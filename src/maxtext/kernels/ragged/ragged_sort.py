# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ragged token sorting operations with custom VJP."""

import jax
import jax.numpy as jnp
from maxtext.kernels.ragged.ragged_gather import ragged_gather
from maxtext.kernels.ragged.ragged_gather_reduce_v2 import ragged_gather_reduce


def ring_ragged_sort(
    hidden_states_local,
    topk_indices_local,
    num_experts,
    topk,
    ep_name,
    ep_size,
    buffer_size=None,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
    use_single_sparsecore=False,
    tag_routing_fn=None,
):
  """Ragged-gather variant for AG-RS Expert Parallelism token routing.

  Unlike :func:`a2a_ragged_sort`, which operates on a valid prefix within a single shard,
  this function sorts and gathers tokens across the global expert space but extracts
  only the specific range of output rows assigned to the experts residing on this shard.
  The rest of the shard's output buffer is padded with zeros.

  Forward:
    Sorts tokens based on ``topk_indices_local`` and gathers
    ``hidden_states_local[token_indices_sorted[i]]`` into ``out[i]`` for
    ``i`` in ``[shard_output_start, shard_output_end)``; other rows are zeroed.

  Backward (gather-reduce):
    ``g_hidden_states[token_indices_sorted[i]] += g_out[i]`` for
    ``i`` in ``[shard_output_start, shard_output_end)``. Because each input token
    contributes to exactly ``topk`` experts, this maps cleanly to a
    ``ragged_gather_reduce`` with ``reduce_group_size=topk`` along the inverse
    permutation.

  Args:
    hidden_states_local: 2D ``[num_tokens_local, hidden]`` input tensor.
    topk_indices_local: 2D ``[num_tokens_local, topk]`` tensor of target expert indices.
    num_experts: scalar ``int`` representing the total global number of experts.
    topk: scalar ``int`` representing the routing top-k factor.
    ep_name: ``str`` identifying the expert parallel axis name.
    ep_size: scalar ``int`` representing the expert parallel mesh size.
    buffer_size: optional scalar ``int`` representing the size of the local buffer.
    tag_routing_fn: optional function applied to each integer routing array (the sort permutation, its inverse,
      the sorted token indices and the group sizes).

  Returns:
    A tuple containing:
      - Processed activations tensor of shape ``[buffer_size, hidden]``
      containing
        only the tokens destined for local experts, padded with zeros elsewhere.
      - 1D tensor ``group_sizes_local`` tracking expert token counts.
      - 1D tensor ``topk_argsort_revert_indices`` for inverse routing.
      - 1D tensor ``topk_argsort_indices``, the inverse of ``topk_argsort_revert_indices``.
  """
  # The routing index maps depend only on the (integer) expert ids, so they are computed outside the custom_vjp.
  # Only the gather below is differentiated.
  num_tokens_local = hidden_states_local.shape[0]

  topk_indices_flat = topk_indices_local.flatten()  # num_tokens_local x topk
  topk_argsort_indices = jnp.argsort(topk_indices_flat)  # num_tokens_local x topk

  token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)  # num_tokens_local x topk
  token_indices_sorted = token_indices[topk_argsort_indices]  # num_tokens_local x topk

  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)  # GLOBAL_NUM_EXPERTS

  topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)  # num_tokens_local x topk

  if tag_routing_fn is not None:
    topk_argsort_indices, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices = (
        tag_routing_fn(t)
        for t in (topk_argsort_indices, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices)
    )

  @jax.custom_vjp
  def _ring_ragged_sort(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices):
    """Gather activations to different EP shards."""
    return _ring_ragged_sort_fwd(
        hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices
    )[0]

  @jax.named_scope("ragged-sort-fwd")
  def _ring_ragged_sort_fwd(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices):
    """Gather activations forward pass."""

    num_tokens_local = hidden_states_local.shape[0]
    shard_idx = jax.lax.axis_index(ep_name)

    local_num_experts = num_experts // ep_size
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    if buffer_size is None or buffer_size >= num_tokens_local * topk:
      local_buffer_size = num_tokens_local * topk
      x = ragged_gather(
          hidden_states_local,
          token_indices_sorted,
          shard_output_start[None],
          shard_output_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
    else:
      local_buffer_size = buffer_size
      # We only gather up to the available buffer size or the actual number of
      # tokens destined for this shard's experts, whichever is smaller.
      gather_end = jnp.minimum(shard_output_end - shard_output_start, local_buffer_size)
      # Pad the indices to ensure we can safely slice a block of size `local_buffer_size`
      # starting at `shard_output_start` without going out-of-bounds during compilation.
      padded_token_indices_sorted = jnp.pad(token_indices_sorted, (0, local_buffer_size))
      sliced_indices = jax.lax.dynamic_slice_in_dim(
          padded_token_indices_sorted,
          shard_output_start,
          local_buffer_size,
          axis=0,
      )
      x = ragged_gather(
          hidden_states_local,
          sliced_indices,
          jnp.int32(0)[None],
          gather_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )

    res = (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        local_buffer_size,
        hidden_states_local.shape,
    )

    return x, res

  @jax.named_scope("ragged-sort-bwd")
  def _ring_ragged_sort_bwd(res, g_out):
    """Backward pass for the gather: a Pallas SC ragged gather reduce.
    The forward gathers ``hidden_states_local[token_indices_sorted[i]]`` into
    ``x[i]`` for ``i`` in ``[shard_output_start, shard_output_end)``.  The
    gradient w.r.t. ``hidden_states_local`` is therefore::
        g_hidden_states[token_indices_sorted[i]] += g_x[i]    (i in valid range)
    which is exactly a ragged gather reduce.
    """
    (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        local_buffer_size,
        _,
    ) = res
    g_x = g_out
    # Restrict to the [start, end) source range via a validity bitmask. The
    # ragged kernel packs valid rows to the front of each row-partition and
    # only iterates over the populated prefix, so we hand it the mask directly
    # rather than materializing a (mostly-zero) dense buffer ourselves.
    n = topk_argsort_revert_indices.shape[0]

    if local_buffer_size >= n:
      valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
          topk_argsort_revert_indices < shard_output_end
      )
      # The forward scatter-add over `token_indices_sorted` is equivalent to a
      # gather-reduce: each input token has exactly `topk` contributions located
      # at sorted positions `topk_argsort_revert_indices[t*topk:(t+1)*topk]`.
      # `topk_weights` is set to ones because this op has no per-row weighting.
      grad_hidden_states = ragged_gather_reduce(
          g_x,
          topk_argsort_revert_indices,
          topk_weights=jnp.ones((n,), dtype=jnp.float32),
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
    else:
      # Buffering: g_x has size `local_buffer_size` (packed).
      # The revert indices are global [0, n), but they must map to the local
      # packed g_x buffer.
      shifted_indices = topk_argsort_revert_indices - shard_output_start
      local_num_tokens = shard_output_end - shard_output_start
      # We only reduce gradients from the valid portion of the local buffer.
      limit = jnp.minimum(local_num_tokens, local_buffer_size)
      # Mask out tokens that were not gathered (either because they belong to
      # other shards, or they exceeded the local buffer size).
      valid_rows_mask = (shifted_indices >= 0) & (shifted_indices < limit)
      # Clamp invalid indices to 0 to prevent compile-time/run-time out-of-bounds
      # in JAX. These clamped values will be ignored due to `valid_rows_mask`.
      safe_indices = jnp.where(valid_rows_mask, shifted_indices, 0)

      grad_hidden_states = ragged_gather_reduce(
          g_x,
          safe_indices,
          topk_weights=jnp.ones((n,), dtype=jnp.float32),
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
    return grad_hidden_states, None, None, None

  _ring_ragged_sort.defvjp(_ring_ragged_sort_fwd, _ring_ragged_sort_bwd)

  x = _ring_ragged_sort(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices)
  return x, group_sizes_local, topk_argsort_revert_indices, topk_argsort_indices


def _unsort_kwargs(
    enforce_gather_fallback,
    enforce_gather_reduce_fallback,
    gather_flops_override,
    gather_reduce_flops_override,
    gather_bytes_accessed_override,
    gather_reduce_bytes_accessed_override,
    use_single_sparsecore,
):
  """The keyword arguments of the unsort's ragged gather (backward) and ragged gather-reduce (forward)."""
  gather_kwargs = {
      "enforce_fallback": enforce_gather_fallback,
      "flops_override": gather_flops_override,
      "bytes_accessed_override": gather_bytes_accessed_override,
      "use_single_sparsecore": use_single_sparsecore,
  }
  gather_reduce_kwargs = {
      "enforce_fallback": enforce_gather_reduce_fallback,
      "flops_override": gather_reduce_flops_override,
      "bytes_accessed_override": gather_reduce_bytes_accessed_override,
      "use_single_sparsecore": use_single_sparsecore,
  }
  return gather_kwargs, gather_reduce_kwargs


def _ring_unsort_window(group_sizes_local, local_num_experts, ep_name):
  """This shard's ``[start, end)`` window in the globally expert-sorted slot order."""
  group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)

  shard_idx = jax.lax.axis_index(ep_name)
  experts_start = shard_idx * local_num_experts
  experts_end = experts_start + local_num_experts

  return group_offsets[experts_start], group_offsets[experts_end]


def _ring_unsort_combine(
    sorted_tokens_local,
    topk_argsort_revert_indices,
    topk_weights_flat,
    shard_output_start,
    shard_output_end,
    topk,
    gather_reduce_kwargs,
):
  """The unsort's forward: each token's routing-weighted sum over its slots on this shard's experts."""
  buffer_size = sorted_tokens_local.shape[0]
  num_tokens = topk_argsort_revert_indices.shape[0]

  # We support two buffering modes:
  # 1. buffer_size >= num_tokens: sorted_tokens_local has size >= num_tokens,
  #    and local tokens are at their global positions [start, end).
  # 2. buffer_size < num_tokens: sorted_tokens_local has size < num_tokens,
  #    and local tokens are packed at [0, local_num_tokens).
  if buffer_size >= num_tokens:
    # Express the scatter as a gather-reduce: each output row pulls
    # from sorted_tokens_local at position `topk_argsort_revert_indices[i]` if
    # that position is within this shard's [start, end) range, else zero.
    # The routing weights are applied per-row before the topk reduction.
    valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
        topk_argsort_revert_indices < shard_output_end
    )
    return ragged_gather_reduce(
        sorted_tokens_local,
        topk_argsort_revert_indices,
        topk_weights=topk_weights_flat,
        valid_rows_mask=valid_rows_mask,
        reduce_group_size=topk,
        **gather_reduce_kwargs,
    )
  # Shift indices so they map to the packed local buffer [0, local_num_tokens).
  shifted_indices = topk_argsort_revert_indices - shard_output_start
  local_num_tokens = shard_output_end - shard_output_start
  limit = jnp.minimum(local_num_tokens, buffer_size)
  valid_rows_mask = (shifted_indices >= 0) & (shifted_indices < limit)
  safe_indices = jnp.where(valid_rows_mask, shifted_indices, 0)
  return ragged_gather_reduce(
      sorted_tokens_local,
      safe_indices,
      topk_weights=topk_weights_flat,
      valid_rows_mask=valid_rows_mask,
      reduce_group_size=topk,
      **gather_reduce_kwargs,
  )


def _ring_unsort_fan_out(
    g_hidden_states_local,
    topk_argsort_revert_indices,
    topk_weights_flat,
    shard_output_start,
    shard_output_end,
    buffer_size,
    topk,
    topk_argsort_indices,
    gather_kwargs,
    mask_rows=True,
):
  """The first half of the unsort's backward: the output cotangent sent to the rows of the sorted buffer.

  Returns ``(gathered, weight_for_sorted, mask, to_slot_order)``: row ``j`` of ``gathered`` is the unweighted
  cotangent of the token that sorted row ``j`` came from, zero outside ``mask`` (with ``mask_rows=True``); ``weight_for_sorted[j]`` is that
  slot's routing weight; ``mask`` marks the rows of this shard's experts (and, for a packed buffer, below its row
  count); ``to_slot_order`` maps a per-row float32 value that is zero outside ``mask`` back to flat slot order, with
  zero for the slots of other shards and for dropped slots.

  With ``mask_rows=False`` the rows of ``gathered`` outside ``mask`` are left as the gather wrote them (on the
  SparseCore kernel: not written, i.e. arbitrary values, possibly NaN); only a caller whose every reader of those
  rows ignores or masks them may pass it.
  """
  n = topk_argsort_revert_indices.shape[0]
  # Build the inverse permutation idx_inv such that idx_inv[j] = i
  # where revert[i] = j. revert is itself argsort(topk_argsort_indices), so its inverse is
  # topk_argsort_indices; reuse it when the caller has it rather than sorting again.
  if topk_argsort_indices is not None:
    idx_inv = topk_argsort_indices
  else:
    idx_inv = jnp.argsort(topk_argsort_revert_indices)

  # Handle the same two buffering modes for backward pass.
  # ragged_gather does the fan-out, by indexing into the un-expanded
  # g_hidden_states_local via idx_inv // topk. It gathers unweighted so the same rows
  # feed both gradients: the activation one after scaling, the weight one after a dot.
  if buffer_size >= n:
    weight_for_sorted = topk_weights_flat[idx_inv]
    gathered = ragged_gather(
        g_hidden_states_local,
        idx_inv // topk,
        shard_output_start[None],
        shard_output_end[None],
        **gather_kwargs,
    )
    # Mask out gradients that correspond to elements outside the valid shard
    # output range.
    mask = (jnp.arange(n) >= shard_output_start) & (jnp.arange(n) < shard_output_end)
    if mask_rows:
      gathered = jnp.where(mask[:, None], gathered, 0.0)

    def from_buffer_rows(per_row):
      # Row j of the sorted buffer is slot idx_inv[j], so slot i reads row revert[i].
      return per_row[topk_argsort_revert_indices]

    return gathered, weight_for_sorted, mask, from_buffer_rows

  # Slice the inverse permutation to match the packed local buffer.
  padded_idx_inv = jnp.pad(idx_inv, (0, buffer_size))
  sliced_idx_inv = jax.lax.dynamic_slice_in_dim(padded_idx_inv, shard_output_start, buffer_size, axis=0)
  gather_end = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
  # Slice the per-slot routing weights to match the packed local buffer.
  padded_weights = jnp.pad(topk_weights_flat[idx_inv], (0, buffer_size))
  sliced_weights = jax.lax.dynamic_slice_in_dim(padded_weights, shard_output_start, buffer_size, axis=0)
  gathered = ragged_gather(
      g_hidden_states_local,
      sliced_idx_inv // topk,
      jnp.int32(0)[None],
      gather_end[None],
      **gather_kwargs,
  )
  # Mask out gradients for elements beyond the valid limit of the local buffer.
  limit = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
  mask = jnp.arange(buffer_size) < limit
  if mask_rows:
    gathered = jnp.where(mask[:, None], gathered, 0.0)

  def from_packed_rows(per_row):
    # Scatter the per-row value back to flat slot order; dropped slots stay zero.
    slots = jnp.where(mask, sliced_idx_inv, n)
    return jnp.zeros((n,), jnp.float32).at[slots].set(per_row, mode="drop")

  return gathered, sliced_weights, mask, from_packed_rows


def ring_ragged_unsort(
    sorted_tokens_local,
    group_sizes_local,
    topk_argsort_revert_indices,
    topk,
    local_num_experts,
    ep_name,
    topk_weights,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
    use_single_sparsecore=False,
    topk_argsort_indices=None,
):
  """Dual of :func:`ring_ragged_sort`.

  Forward:
    ``out[i] = sum_k( w[i*topk+k] * sorted_tokens_local[topk_argsort_revert_indices[i*topk+k]] )``
    for positions where ``topk_argsort_revert_indices`` is in
    ``[shard_output_start, shard_output_end)``; other rows are zeroed.
    This scatters the processed outputs from the experts hosted on this shard
    back to their flat arrival buffer positions, applying routing weights
    during the reduction.

  Backward:
    ``g_sorted_tokens[j] = w[i] * g_out[i // topk]`` and
    ``g_w[i] = <g_out[i // topk], sorted_tokens_local[j]>`` where
    ``j = topk_argsort_revert_indices[i]`` and ``j`` is in
    ``[shard_output_start, shard_output_end)``.

  Args:
    sorted_tokens_local: 2D ``[buffer_size, hidden]`` output tensor from local experts.
    group_sizes_local: 1D tensor tracking the token loads per expert.
    topk_argsort_revert_indices: 1D permutation restoring flat token positions.
    topk: scalar ``int`` representing the routing top-k factor.
    local_num_experts: scalar ``int`` representing the count of experts hosted on this shard.
    ep_name: ``str`` identifying the expert parallel axis name.
    topk_weights: ``[num_tokens_local * topk]`` tensor of per-slot routing weights.
      Differentiated: its gradient is what trains the router.
    topk_argsort_indices: optional inverse of ``topk_argsort_revert_indices`` (as returned by
      :func:`ring_ragged_sort`). The backward needs this permutation; when it is
      None, the backward re-derives it with an argsort. Both give the same permutation.

  Returns:
    A 2D ``[num_tokens_local, hidden]`` tensor with expert outputs scattered back
    to their original global sequence locations, with unpopulated positions zeroed out.
  """
  gather_kwargs, gather_reduce_kwargs = _unsort_kwargs(
      enforce_gather_fallback,
      enforce_gather_reduce_fallback,
      gather_flops_override,
      gather_reduce_flops_override,
      gather_bytes_accessed_override,
      gather_reduce_bytes_accessed_override,
      use_single_sparsecore,
  )

  @jax.custom_vjp
  def _ring_ragged_unsort(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      topk_argsort_indices,
  ):
    """Unsort and scatter activations."""
    return _ring_ragged_unsort_fwd(
        sorted_tokens_local,
        group_sizes_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        topk_argsort_indices,
    )[0]

  @jax.named_scope("ragged-unsort-fwd")
  def _ring_ragged_unsort_fwd(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      topk_argsort_indices,
  ):
    """Executes unsorting sending tokens back."""
    shard_output_start, shard_output_end = _ring_unsort_window(group_sizes_local, local_num_experts, ep_name)
    out = _ring_unsort_combine(
        sorted_tokens_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        topk,
        gather_reduce_kwargs,
    )
    res = (
        sorted_tokens_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        sorted_tokens_local.shape[0],
        topk_argsort_indices,
    )
    return out, res

  @jax.named_scope("ragged-unsort-bwd")
  def _ring_ragged_unsort_bwd(res, g_out):
    """Backward pass for the scatter with routing weights.

    The forward computes (per output token t, with topk slots k=0..topk-1):
      out[t] = sum_k  w[t*topk+k] * sorted_tokens[revert[t*topk+k]]
    masked by validity.

    Gradient w.r.t. sorted_tokens:
      g_sorted_tokens[j] = w[i] * g_out[i // topk]
    where j = revert[i] and j in [start, end).

    Gradient w.r.t. the routing weights:
      g_w[i] = <g_out[i // topk], sorted_tokens[revert[i]]>
    for valid i, zero otherwise.
    """
    (
        sorted_tokens_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
        topk_argsort_indices,
    ) = res
    n = topk_argsort_revert_indices.shape[0]
    gathered, weight_for_sorted, _, to_slot_order = _ring_unsort_fan_out(
        g_out,
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
        topk,
        topk_argsort_indices,
        gather_kwargs,
    )
    grad_sorted_tokens = (gathered * weight_for_sorted[:, None]).astype(gathered.dtype)
    # Row-wise dot in sorted order, then back to flat slot order (a full buffer has its rows at [0, n)).
    sorted_rows = sorted_tokens_local[:n] if buffer_size >= n else sorted_tokens_local
    dot = jnp.sum(gathered.astype(jnp.float32) * sorted_rows.astype(jnp.float32), axis=-1)
    grad_topk_weights = to_slot_order(dot)
    return grad_sorted_tokens, None, None, grad_topk_weights, None

  _ring_ragged_unsort.defvjp(_ring_ragged_unsort_fwd, _ring_ragged_unsort_bwd)

  # Build the flat weights array from the routing weights.
  topk_weights_flat = topk_weights.astype(jnp.float32)

  return _ring_ragged_unsort(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      topk_argsort_indices,
  )


def ring_ragged_unsort_down_projection(
    intermediate_local,
    wo_local,
    down_projection,
    down_projection_operands,
    group_sizes_local,
    topk_argsort_revert_indices,
    topk,
    local_num_experts,
    ep_name,
    topk_weights,
    topk_argsort_indices=None,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
    use_single_sparsecore=False,
    mask_fan_out_rows=True,
):
  """:func:`ring_ragged_unsort` of the down projection, with a backward that does not read its output.

  Forward: ``y = down_projection(intermediate_local, wo_local, *down_projection_operands)``, the per-expert GEMM
  ``y_j = h_j @ wo[e(j)]`` over the sorted rows ``h_j``, then the forward of ``ring_ragged_unsort(y, ...)`` through
  the same helpers (``_ring_unsort_window``, ``_ring_unsort_combine``), so the output is bit-identical to the two
  ops apart. The backward's fan-out is ``ring_ragged_unsort``'s too (``_ring_unsort_fan_out``).

  Backward. Let ``g_j`` be the unweighted output cotangent of the token that sorted row ``j`` came from and ``w_j``
  that slot's routing weight. The two ops apart compute ``dy_j = w_j g_j``, ``dh = dy @ wo^T``, ``dwo = h^T @ dy``
  and the routing-weight gradient ``g_w_j = <g_j, y_j>``. That last dot reads ``y``, so a rematerialized layer
  recomputes the down-projection GEMM in the backward only to form it. Row ``j`` of the GEMM is linear in ``h_j``
  and the weight is a scalar per row, so::

      <g_j, h_j wo> = <g_j wo^T, h_j>,   (w_j g_j) wo^T = w_j (g_j wo^T),   h^T (w * g) = (w * h)^T g.

  This backward therefore takes the down projection's own VJP at the input ``w * h`` with the cotangent ``g``:
  ``u = g wo^T`` and ``dwo = (w * h)^T g`` (the GEMM's own backward kernels), then::

      dh_j = w_j u_j,     g_w_j = <u_j, h_j>   (float32).

  The GEMMs keep their shapes; ``y`` is not a residual, so XLA drops its recomputation. Each gradient still has
  one rounding to the activation dtype, but at a different point (``w * h`` instead of ``w * g``; ``dh`` after the
  product with ``w`` instead of before the GEMM; ``g_w`` from ``g wo^T`` instead of ``y``), so the gradients agree
  with the two ops apart to that rounding, not bit for bit.

  Args:
    intermediate_local: ``[buffer_size, mlp]`` down-projection input, rows grouped by local expert.
    wo_local: the down-projection kernel of the local experts, as ``down_projection`` takes it.
    down_projection: ``(h, wo, *down_projection_operands) -> [buffer_size, hidden]``, a GEMM with its own VJP. It
      must not close over traced arrays; pass them in ``down_projection_operands``.
    down_projection_operands: tuple of non-differentiated operands of ``down_projection`` (group sizes and the
      like).
    group_sizes_local, topk_argsort_revert_indices, topk, local_num_experts, ep_name, topk_weights,
      topk_argsort_indices: as for :func:`ring_ragged_unsort`.
    mask_fan_out_rows: zero the fanned-out cotangent ``g`` outside this shard's rows before the down projection's
      VJP (a ``[rows, hidden]`` pass). ``False`` skips it: those rows then hold whatever the gather left (arbitrary,
      possibly NaN, on the SparseCore kernel), so only pass it when ``down_projection``'s VJP reads no row outside
      its groups into a valid result -- its input gradient is row-wise and its weight gradient masks rows outside
      the groups on both operands (the gmm_v2 tgmm kernel does). ``dh`` and the routing-weight gradient are masked
      on the ``[rows, mlp]`` side either way.

  Returns:
    The unsorted ``[num_tokens_local, hidden]`` output.
  """
  num_slots = topk_argsort_revert_indices.shape[0]
  if intermediate_local.shape[0] > num_slots:
    # ring_ragged_sort's buffer has exactly num_slots rows (full mode) or fewer (packed mode).
    raise ValueError(
        f"ring_ragged_unsort_down_projection: {intermediate_local.shape[0]} buffer rows > {num_slots} slots."
    )
  gather_kwargs, gather_reduce_kwargs = _unsort_kwargs(
      enforce_gather_fallback,
      enforce_gather_reduce_fallback,
      gather_flops_override,
      gather_reduce_flops_override,
      gather_bytes_accessed_override,
      gather_reduce_bytes_accessed_override,
      use_single_sparsecore,
  )

  @jax.custom_vjp
  def _unsort_down_projection(h, wo, operands, group_sizes, revert, weights, argsort_indices):
    """Down projection, then unsort and combine."""
    return _unsort_down_projection_fwd(h, wo, operands, group_sizes, revert, weights, argsort_indices)[0]

  @jax.named_scope("ragged-unsort-down-projection-fwd")
  def _unsort_down_projection_fwd(h, wo, operands, group_sizes, revert, weights, argsort_indices):
    y = down_projection(h, wo, *operands)
    start, end = _ring_unsort_window(group_sizes, local_num_experts, ep_name)
    out = _ring_unsort_combine(y, revert, weights, start, end, topk, gather_reduce_kwargs)
    # y is not a residual: the backward below never reads it.
    res = (h, wo, operands, revert, weights, start, end, argsort_indices)
    return out, res

  @jax.named_scope("ragged-unsort-down-projection-bwd")
  def _unsort_down_projection_bwd(res, g_out):
    h, wo, operands, revert, weights, start, end, argsort_indices = res
    gathered, weight_for_sorted, mask, to_slot_order = _ring_unsort_fan_out(
        g_out,
        revert,
        weights,
        start,
        end,
        h.shape[0],
        topk,
        argsort_indices,
        gather_kwargs,
        mask_rows=mask_fan_out_rows,
    )
    weight = weight_for_sorted[:, None]
    h32 = h.astype(jnp.float32)
    # The routing weight goes on the narrow [rows, mlp] side: dwo = (w * h)^T g.
    weighted_h = jnp.where(mask[:, None], h32 * weight, 0.0).astype(h.dtype)
    # The down projection's own VJP at the input w * h: its input gradient g wo^T does not depend on the input,
    # and its weight gradient is (w * h)^T g. Its forward GEMM has no reader, so XLA removes it.
    y_unused, project_vjp = jax.vjp(lambda h_, wo_: down_projection(h_, wo_, *operands), weighted_h, wo)
    u, grad_wo = project_vjp(gathered.astype(y_unused.dtype))
    u = u.astype(jnp.float32)
    grad_h = jnp.where(mask[:, None], u * weight, 0.0).astype(h.dtype)
    # The routing-weight gradient <g, h wo> = <g wo^T, h>, from the down projection's input.
    dot = jnp.where(mask, jnp.sum(u * h32, axis=-1), 0.0)
    grad_topk_weights = to_slot_order(dot)
    grad_operands = jax.tree.map(lambda _: None, operands)
    return grad_h, grad_wo, grad_operands, None, None, grad_topk_weights, None

  _unsort_down_projection.defvjp(_unsort_down_projection_fwd, _unsort_down_projection_bwd)

  return _unsort_down_projection(
      intermediate_local,
      wo_local,
      tuple(down_projection_operands),
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights.astype(jnp.float32),
      topk_argsort_indices,
  )


def a2a_ragged_sort(
    inputs,
    sort_indices,
    valid_end,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    use_single_sparsecore=False,
):
  """Ragged-gather variant for ``local_permute``.

  Unlike :func:`ring_ragged_sort`, the rows valid for this shard live in
  the prefix ``[0, valid_end)`` of ``inputs`` (the rest of the buffer is
  padding). This helper sorts ``inputs`` by ``sort_indices`` but only touches
  the valid prefix, making the cost proportional to the *actual* token count
  rather than the padded buffer length.

  Forward:
    ``out[i] = inputs[sort_indices[i]]`` for ``i in [0, valid_end)``;
    other rows are zero.

  Backward (gather-reduce):
    ``g_inputs[sort_indices[i]] += g_out[i]`` for ``i in [0, valid_end)``.
    Because ``sort_indices`` is a permutation, each input row receives exactly
    one contribution; we model this as a ``ragged_gather_reduce`` with
    ``reduce_group_size=1`` along the inverse permutation.

  Args:
    inputs: 2D ``[num_tokens, hidden]`` tensor whose valid rows live in the
      prefix ``[0, valid_end)``.
    sort_indices: 1D permutation of ``[0, num_tokens)`` describing the desired
      ordering. Values at positions ``>= valid_end`` are ignored.
    valid_end: scalar ``int32`` indicating the exclusive end of the valid
      prefix.

  Returns:
    A 2D ``[num_tokens, hidden]`` tensor sorted by ``sort_indices`` over the
    valid prefix, with padded rows zeroed.
  """

  @jax.custom_vjp
  def _a2a_ragged_sort(inputs, sort_indices, valid_end):
    return _a2a_ragged_sort_fwd(inputs, sort_indices, valid_end)[0]

  @jax.named_scope("local-ragged-sort-fwd")
  def _a2a_ragged_sort_fwd(inputs, sort_indices, valid_end):
    start = jnp.int32(0)
    end = valid_end.astype(jnp.int32) if hasattr(valid_end, "astype") else jnp.int32(valid_end)
    out = ragged_gather(
        inputs,
        sort_indices,
        start[None],
        end[None],
        use_single_sparsecore=use_single_sparsecore,
    )
    n = sort_indices.shape[0]
    valid_mask = jnp.arange(n) < end
    out = jnp.where(valid_mask[:, None], out, 0.0)
    res = (sort_indices, end, inputs.shape)
    return out, res

  @jax.named_scope("local-ragged-sort-bwd")
  def _a2a_ragged_sort_bwd(res, g_out):
    sort_indices, end, _ = res
    n = sort_indices.shape[0]
    valid_rows_mask = jnp.arange(n) < end
    # g_inputs[sort_indices[i]] += g_out[i], for i in [0, end). This is a
    # ragged scatter-add, which we express as a gather-reduce along the inverse
    # permutation: each input row j receives exactly one contribution from
    # output row i where sort_indices[i] == j.
    idx_inv = jnp.argsort(sort_indices)
    grad_inputs = ragged_gather_reduce(
        g_out,
        idx_inv,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask[idx_inv],
        reduce_group_size=1,
        enforce_fallback=enforce_gather_reduce_fallback,
        use_single_sparsecore=use_single_sparsecore,
    )
    # custom_vjp must return one gradient per primal arg; valid_end is integer
    # and non-differentiable, so we return None for it.
    return grad_inputs, None, None

  _a2a_ragged_sort.defvjp(_a2a_ragged_sort_fwd, _a2a_ragged_sort_bwd)
  return _a2a_ragged_sort(inputs, sort_indices, valid_end)


def a2a_ragged_unsort(
    sorted_tokens,
    revert_indices,
    valid_end,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    use_single_sparsecore=False,
):
  """Dual of :func:`a2a_ragged_sort`.

  Forward:
    ``out[i] = sorted_tokens[revert_indices[i]]`` for ``i in [0, valid_end)``;
    other rows are zero. This is the unsort step in the local-permute path:
    given a buffer ordered by local expert IDs, restore the original arrival
    order.

  Backward:
    ``g_sorted_tokens[j] = g_out[i]`` where ``j = revert_indices[i]``, for
    ``i in [0, valid_end)``. Since ``revert_indices`` is a permutation, this
    is a simple ragged gather over the inverse permutation.

  Args:
    sorted_tokens: 2D ``[num_tokens, hidden]`` tensor.
    revert_indices: 1D permutation of ``[0, num_tokens)``.
    valid_end: scalar ``int32`` indicating the exclusive end of the valid
      prefix.

  Returns:
    A 2D ``[num_tokens, hidden]`` tensor with rows reordered by
    ``revert_indices`` over the valid prefix and zero elsewhere.
  """

  @jax.custom_vjp
  def _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end):
    return _a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end)[0]

  @jax.named_scope("local-ragged-unsort-fwd")
  def _a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end):
    start = jnp.int32(0)
    end = valid_end.astype(jnp.int32) if hasattr(valid_end, "astype") else jnp.int32(valid_end)
    n = revert_indices.shape[0]
    valid_rows_mask = jnp.arange(n) < end
    out = ragged_gather_reduce(
        sorted_tokens,
        revert_indices,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask,
        reduce_group_size=1,
        enforce_fallback=enforce_gather_reduce_fallback,
        use_single_sparsecore=use_single_sparsecore,
    )
    res = (revert_indices, end, sorted_tokens.shape, start)
    return out, res

  @jax.named_scope("local-ragged-unsort-bwd")
  def _a2a_ragged_unsort_bwd(res, g_out):
    revert_indices, end, sorted_tokens_shape, start = res
    # g_sorted_tokens[revert_indices[i]] = g_out[i] for i in [0, end).
    # Because revert_indices is a permutation, build the inverse and use
    # ragged_gather to pull the per-row gradients to the right positions.
    idx_inv = jnp.argsort(revert_indices)
    grad_sorted = ragged_gather(
        g_out,
        idx_inv,
        start[None],
        end[None],
        use_single_sparsecore=use_single_sparsecore,
    )
    num_rows = sorted_tokens_shape[0]
    pos = jnp.arange(num_rows)
    valid = pos < end
    grad_sorted = jnp.where(valid[:, None], grad_sorted, jnp.zeros_like(grad_sorted))

    return grad_sorted, None, None

  _a2a_ragged_unsort.defvjp(_a2a_ragged_unsort_fwd, _a2a_ragged_unsort_bwd)
  return _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end)
