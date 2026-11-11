# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""moe_router_grad_from_wo_input: the wo GMM and the unsort as one op, against the two apart, on 8 CPU devices.

`ragged_sort.ring_ragged_unsort_down_projection` runs the down-projection GEMM and then the forward of
`ring_ragged_unsort` (the same helpers), so the output must be bit-identical. Its backward does not read the GEMM's
output: it takes the GEMM's own VJP at the input w * h with the unweighted cotangent g, giving u = g wo^T and
dwo = (w * h)^T g, then dh = w * u and the routing-weight gradient <u, h> (= <g, h wo>). Each gradient keeps one
rounding to the activation dtype, at a different point, so the gradients agree with the two ops apart to bf16
rounding, not bit for bit.

What a CPU run reaches: the ragged gather and gather-reduce take their jnp reference fallbacks off a TPU (the
SparseCore kernels are TPU-only) and the layer's GMM is megablox in interpret mode. The fused op's own math (the
custom VJP, the weighting, the dot, the masks, the slot-order return in both buffering modes) is plain JAX and runs
here as on TPU.

Pre-registered (thresholds from the upstream MLPerf test this file is cut down from):
  - The op on a 4-device expert-parallel mesh, a jnp grouped matmul as the down projection, full and packed (token
    dropping) buffer: output bit-identical to the two ops apart, dx / dw / dwi / dwo within relative L2 1e-2.
  - Precision, the op alone (h an input, so only the backward rounds): against the function written densely in
    float32, each fused gradient (dh, dw, dwo) is no further than 1.10 times as far as the two-ops gradient, over 6
    seeds x {full, packed}. The routing weights sit in [0.25, 0.3), the bottom of a binade, where bf16 rounds them
    worst, so a backward that rounds them is visible (upstream measured 1.31-1.34 on dh and dwo for that defect).
    Both paths share the unsort helpers, so each path's error must also stay within 1e-2 of the float32 reference.
  - The Hy4-tiny routed MoE (8 experts, top 2, EP 4 x FSDP 2, ragged sort, megablox interpret): full buffer, packed
    buffer, num_moe_token_chunks=2, prefuse_moe_weights: output and loss bit-identical to the key off, every gradient
    within relative L2 1e-2, and the compiled jax.grad(jax.checkpoint(loss)) has exactly num_moe_token_chunks GEMM
    loops fewer (the wo GEMM is no longer recomputed in the backward).
  - The validator rejects the key for each requirement it does not meet, naming it.
  - mask_fan_out_rows=False (added later; the gmm_v2 path): with NaN injected into the unmasked fan-out rows outside
    the shard's range (full buffer; the packed dropping buffer has none), output bit-identical and dx / dw / dwi /
    dwo finite and within relative L2 1e-2 of the two ops apart, for the grouped matmul and for a double of the
    kernels' worst case (input gradient for every row, weight gradient masked on both operands); positive control:
    a projection whose weight gradient reads every row gets NaN in dwo without the mask and not with it. The real
    kernels' property is checked on TPU in gmm_v2_row_isolation_test.py.
"""

import re
import sys

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.ragged import ragged_sort
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_GRAD_REL_L2 = 1e-2
# The fused backward's error against a float32 run, over the two-ops backward's error.
_PRECISION_RATIO = 1.10
# The largest relative L2 error either path's output or gradients may have against the float32 run.
_PRECISION_MAX_ERR = 1e-2
_PRECISION_SEEDS = range(6)

pytestmark = pytest.mark.skipif(
    jax.device_count() < _NUM_DEVICES,
    reason=f"needs {_NUM_DEVICES} devices (XLA_FLAGS=--xla_force_host_platform_device_count={_NUM_DEVICES})",
)


def _rel(value, reference):
  """Relative L2 distance of `value` from `reference`."""
  return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


# --------------------------------------------------------------------------------------------------- the op alone
_OP_EP, _OP_EXPERTS, _OP_TOPK, _OP_TOKENS, _OP_D, _OP_F = 4, 8, 2, 64, 128, 64
# A packed buffer smaller than the busiest shard's share of the 128 slots (32 on average), so tokens are dropped.
_OP_DROPPING_BUFFER = 24
# The precision check's op: more slots (1024, 256 per shard on average), so each error estimate is tight.
_PRECISION_TOKENS = 512
_PRECISION_DROPPING_BUFFER = 192


def _local_expert(h, group_sizes, group_offset):
  """The local expert of each buffer row (row r is in global group k when ends[k-1] <= r < ends[k])."""
  ends = jnp.cumsum(group_sizes)
  return jnp.sum(jnp.arange(h.shape[0])[:, None] >= ends[None, :], axis=1) - group_offset


def _grouped_matmul(h, w, group_sizes, expert_assignments, group_offset):
  """A down projection with its own (autodiff) VJP: row r uses expert (global group of r) - group_offset."""
  del expert_assignments
  local = _local_expert(h, group_sizes, group_offset)
  out = jnp.zeros((h.shape[0], w.shape[-1]), jnp.float32)
  for e in range(w.shape[0]):
    out = out + jnp.where((local == e)[:, None], h.astype(jnp.float32) @ w[e].astype(jnp.float32), 0.0)
  return out.astype(h.dtype)


def _op_inputs(seed=0, num_tokens=_OP_TOKENS):
  """Tokens, top-k expert ids, routing weights, wi, wo, the output cotangent and per-shard down-projection inputs."""
  k = jax.random.split(jax.random.PRNGKey(seed), 7)
  x = jax.random.normal(k[0], (num_tokens, _OP_D), jnp.float32)
  ids = jnp.stack([jax.random.permutation(kk, _OP_EXPERTS)[:_OP_TOPK] for kk in jax.random.split(k[1], num_tokens)])
  w = jax.nn.softmax(jax.random.normal(k[2], (num_tokens, _OP_TOPK)), axis=-1).reshape(-1)
  wi = jax.random.normal(k[3], (_OP_EXPERTS, _OP_D, _OP_F)) * 0.1
  wo = jax.random.normal(k[4], (_OP_EXPERTS, _OP_F, _OP_D)) * 0.1
  cot = jax.random.normal(k[5], (num_tokens, _OP_D), jnp.float32)
  # For the precision check: the down-projection input of every slot, on every shard.
  h = jax.random.normal(k[6], (_OP_EP, num_tokens * _OP_TOPK, _OP_F), jnp.float32)
  return x, ids.astype(jnp.int32), w, wi, wo, cot, h


def _expert_operands(gs, buffer_rows, num_slots, local):
  """The grouped matmul's operands for this shard's buffer: (group sizes, None, group offset)."""
  shard = jax.lax.axis_index("ep")
  if buffer_rows < num_slots:
    # Packed buffer: this shard's groups from row 0, clipped to the buffer.
    ends = jnp.minimum(jnp.cumsum(jax.lax.dynamic_slice_in_dim(gs, shard * local, local)), buffer_rows)
    return (ends - jnp.concatenate([jnp.zeros((1,), ends.dtype), ends[:-1]]), None, 0)
  return (gs, None, shard * local)


def _op_run(buffer_size, fused, mask_fan_out_rows=True, down_projection=None):
  """bf16 run on a 4-device EP mesh; returns (output, dx, dw, dwi, dwo) summed / gathered over the shards."""
  down_projection = down_projection or _grouped_matmul
  x, ids, w, wi, wo, cot, _ = _op_inputs()
  ep, local = _OP_EP, _OP_EXPERTS // _OP_EP
  mesh = Mesh(np.array(jax.devices()[:ep]), ("ep",))
  wi_s = wi.reshape(ep, local, _OP_D, _OP_F).astype(jnp.bfloat16)
  wo_s = wo.reshape(ep, local, _OP_F, _OP_D).astype(jnp.bfloat16)

  def body(x, w, wi, wo, cot):
    wi, wo = wi[0], wo[0]

    def f(x, w, wi, wo):
      sorted_x, gs, revert, perm = ragged_sort.ring_ragged_sort(
          x, ids, _OP_EXPERTS, _OP_TOPK, "ep", ep, buffer_size=buffer_size
      )
      operands = _expert_operands(gs, sorted_x.shape[0], _OP_TOKENS * _OP_TOPK, local)
      h = jax.nn.silu(_grouped_matmul(sorted_x, wi, *operands).astype(jnp.float32)).astype(sorted_x.dtype)
      if fused:
        return ragged_sort.ring_ragged_unsort_down_projection(
            h,
            wo,
            down_projection,
            operands,
            gs,
            revert,
            _OP_TOPK,
            local,
            "ep",
            topk_weights=w,
            topk_argsort_indices=perm,
            mask_fan_out_rows=mask_fan_out_rows,
        )
      y = _grouped_matmul(h, wo, *operands)
      return ragged_sort.ring_ragged_unsort(
          y, gs, revert, _OP_TOPK, local, "ep", topk_weights=w, topk_argsort_indices=perm
      )

    out, vjp = jax.vjp(f, x, w, wi, wo)
    gx, gw, gwi, gwo = vjp(cot.astype(out.dtype))
    return out[None], gx[None], gw[None], gwi[None], gwo[None]

  sm = jax.shard_map(
      body, mesh=mesh, in_specs=(P(), P(), P("ep"), P("ep"), P()), out_specs=(P("ep"),) * 5, check_vma=False
  )
  outs = [np.asarray(jnp.asarray(a).astype(jnp.float32)) for a in jax.jit(sm)(x.astype(jnp.bfloat16), w, wi_s, wo_s, cot)]
  out, gx, gw, gwi, gwo = outs
  return [
      out.sum(0),
      gx.sum(0),
      gw.sum(0),
      gwi.reshape(_OP_EXPERTS, _OP_D, _OP_F),
      gwo.reshape(_OP_EXPERTS, _OP_F, _OP_D),
  ]


def _dense_unsort_down_projection(h, w, wo, ids, gs, revert, shard, local):
  """The fused op's function written densely, slot by slot (no ragged kernels, no rounding in float32).

  Slot i (token i // top_k) reads buffer row `revert[i]` (packed buffer: minus this shard's window start, and only
  below the buffer's row count), multiplies it by its expert's down projection, and adds it, weighted, to its token.
  """
  num_slots, buffer_rows = revert.shape[0], h.shape[0]
  local_expert = ids.reshape(-1) - shard * local
  is_local = (local_expert >= 0) & (local_expert < local)
  start = jnp.sum(jnp.where(jnp.arange(gs.shape[0]) < shard * local, gs, 0))
  row = revert if buffer_rows >= num_slots else revert - start
  limit = buffer_rows if buffer_rows >= num_slots else jnp.minimum(jnp.sum(jnp.where(is_local, 1, 0)), buffer_rows)
  valid = is_local & (row >= 0) & (row < limit)
  h_slot = jnp.where(valid[:, None], h[jnp.where(valid, row, 0)], 0.0)
  y_slot = jnp.einsum("nf,nfd->nd", h_slot, wo[jnp.clip(local_expert, 0, local - 1)])
  weighted = jnp.where(valid[:, None], (w[:, None] * y_slot), 0.0)
  return jnp.sum(weighted.reshape(num_slots // _OP_TOPK, _OP_TOPK, -1), axis=1)


def _op_backward_run(seed, buffer_size, path):
  """The op (path "fused"), the two ops apart ("two_ops", both bf16) or the dense float32 function ("reference"),
  differentiated at (h, w, wo) with the cotangent g on a 4-device EP mesh. Returns (out, dh, dw, dwo)."""
  x, ids, _, _, wo, cot, h = _op_inputs(seed, _PRECISION_TOKENS)
  w = jax.random.uniform(jax.random.PRNGKey(1000 + seed), (_PRECISION_TOKENS * _OP_TOPK,), jnp.float32, 0.25, 0.3)
  ep, local = _OP_EP, _OP_EXPERTS // _OP_EP
  mesh = Mesh(np.array(jax.devices()[:ep]), ("ep",))
  # The same bf16-representable inputs on every path; the reference only computes in float32.
  dtype = jnp.float32 if path == "reference" else jnp.bfloat16
  wo_s = wo.reshape(ep, local, _OP_F, _OP_D).astype(jnp.bfloat16).astype(dtype)
  h_s = h.astype(jnp.bfloat16).astype(dtype)
  cot = cot.astype(jnp.bfloat16).astype(dtype)

  def body(h, w, wo, cot):
    h, wo = h[0], wo[0]
    sorted_x, gs, revert, perm = ragged_sort.ring_ragged_sort(
        x.astype(jnp.bfloat16), ids, _OP_EXPERTS, _OP_TOPK, "ep", ep, buffer_size=buffer_size
    )
    rows = sorted_x.shape[0]
    operands = _expert_operands(gs, rows, _PRECISION_TOKENS * _OP_TOPK, local)

    def f(h, w, wo):
      if path == "reference":
        return _dense_unsort_down_projection(h, w, wo, ids, gs, revert, jax.lax.axis_index("ep"), local)
      if path == "fused":
        return ragged_sort.ring_ragged_unsort_down_projection(
            h, wo, _grouped_matmul, operands, gs, revert, _OP_TOPK, local, "ep", topk_weights=w, topk_argsort_indices=perm
        )
      y = _grouped_matmul(h, wo, *operands)
      return ragged_sort.ring_ragged_unsort(
          y, gs, revert, _OP_TOPK, local, "ep", topk_weights=w, topk_argsort_indices=perm
      )

    out, vjp = jax.vjp(f, h[:rows], w, wo)
    dh, dw, dwo = vjp(cot.astype(out.dtype))
    return out[None], dh[None], dw[None], dwo[None]

  sm = jax.shard_map(body, mesh=mesh, in_specs=(P("ep"), P(), P("ep"), P()), out_specs=(P("ep"),) * 4, check_vma=False)
  out, dh, dw, dwo = [np.asarray(jnp.asarray(a).astype(jnp.float32)) for a in jax.jit(sm)(h_s, w, wo_s, cot)]
  return out, dh, dw.sum(0), dwo.reshape(_OP_EXPERTS, _OP_F, _OP_D)


class FusedOpTest(parameterized.TestCase):
  """ring_ragged_unsort_down_projection against ring_ragged_unsort of the same GEMM."""

  @parameterized.named_parameters(("full_buffer", None), ("packed_buffer", _OP_DROPPING_BUFFER))
  def test_matches_two_ops(self, buffer_size):
    base = _op_run(buffer_size, fused=False)
    new = _op_run(buffer_size, fused=True)
    np.testing.assert_array_equal(new[0], base[0], err_msg="forward output is not bit-identical")
    for name, a, b in zip(("dx", "dw", "dwi", "dwo"), base[1:], new[1:]):
      self.assertTrue(np.isfinite(b).all(), name)
      self.assertGreater(np.linalg.norm(a), 0.0, name)
      rel = _rel(b, a)
      self.assertLess(rel, _GRAD_REL_L2, f"{name}: rel_l2 vs two ops = {rel:.3e}")

  @parameterized.named_parameters(("full_buffer", None), ("packed_buffer", _PRECISION_DROPPING_BUFFER))
  def test_precision_against_float32(self, buffer_size):
    failures, ratios = [], []
    for seed in _PRECISION_SEEDS:
      reference = _op_backward_run(seed, buffer_size, "reference")
      two_ops = _op_backward_run(seed, buffer_size, "two_ops")
      fused = _op_backward_run(seed, buffer_size, "fused")
      for name, r, a, b in zip(("out", "dh", "dw", "dwo"), reference, two_ops, fused):
        err_two_ops, err_fused = _rel(a, r), _rel(b, r)
        ratio = err_fused / err_two_ops if err_two_ops > 0 else float("inf")
        ratios.append(ratio)
        ok = ratio <= _PRECISION_RATIO and bool(np.isfinite(b).all())
        ok &= max(err_two_ops, err_fused) <= _PRECISION_MAX_ERR
        if not ok:
          failures.append(f"seed {seed} {name}: two ops {err_two_ops:.3e} fused {err_fused:.3e} ratio {ratio:.3f}")
    print(f"precision ratios min {min(ratios):.3f} max {max(ratios):.3f}", file=sys.stderr)
    self.assertEqual(failures, [])


# --------------------------------------------------------------------------------------------------- the layer
_LAYER_BASE = {
    "enable_checkpointing": False,
    "log_config": False,
    "skip_jax_distributed_system": True,
    "model_name": "hy4-tiny",
    "override_model_config": True,
    # hy4-tiny's sparse indexer needs dot_product (or tokamax splash flash, TPU-only); this file runs no attention.
    "attention": "dot_product",
    "max_target_length": 32,
    "max_prefill_predict_length": 32,
    "per_device_batch_size": 1,
    "dtype": "bfloat16",
    "weight_dtype": "float32",
    # megablox runs its Pallas kernels in interpret mode off-TPU; tokamax / gmm_v2 are TPU-only.
    "megablox": True,
    "sparse_matmul": True,
    "use_tokamax_gmm": False,
    "use_gmm_v2": False,
    "use_ring_of_experts": True,
    "use_ragged_sort": True,
    "use_custom_sort_vjp": False,
    "ici_fsdp_parallelism": 2,
    "ici_expert_parallelism": 4,
    "ragged_buffer_factor": -1.0,
}


def _cfg(**overrides):
  kwargs = {"run_name": "moe_router_grad_from_wo_input", **_LAYER_BASE, **overrides}
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _build(cfg, mesh):
  return moe.RoutedMoE(
      config=cfg,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      mesh=mesh,
      kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes=("embed", "mlp"),
      intermediate_dim=cfg.moe_mlp_dim,
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
      rngs=nnx.Rngs(params=0),
  )


def _compiled_gemm_loops(hlo_text):
  """Number of while instructions in a compiled HLO module.

  The GMMs are megablox kernels in interpret mode, each a loop over its tiles; identical GEMMs share one body
  computation, so the loop instructions (one per call), not the dots in the bodies, count the GEMMs.
  """
  return sum(" while(" in line for line in hlo_text.splitlines())


def _run_layer(cfg, mesh, params, x):
  """Loss, output and gradients; the number of GEMM loops in the compiled rematerialized backward."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    graphdef, _, rest = nnx.split(_build(cfg, mesh), nnx.Param, ...)

    def loss_fn(params, x):
      out, lb_loss, _ = nnx.merge(graphdef, params, rest)(x)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      if lb_loss is not None:
        loss = loss + lb_loss.astype(jnp.float32)
      return loss, out

    def linear_loss(params, x):
      # Linear in the layer output, as behind the decoder's residual add: the layer's own backward does not need its
      # output, so only the backward's own reads of the wo GEMM output keep that GEMM in the replay.
      out, _, _ = nnx.merge(graphdef, params, rest)(x)
      return jnp.sum(out.astype(jnp.float32) * jax.random.normal(jax.random.PRNGKey(3), out.shape, jnp.float32))

    remat = jax.grad(jax.checkpoint(linear_loss, policy=jax.checkpoint_policies.nothing_saveable), argnums=(0, 1))
    loops = _compiled_gemm_loops(jax.jit(remat).lower(params, x).compile().as_text())
    (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, x)
  return jax.device_get((loss, out, grads)), loops


class LayerTest(parameterized.TestCase):
  """The Hy4-tiny routed MoE with the key off and on, on the same parameters."""

  @parameterized.named_parameters(
      ("full_buffer", {}),
      ("packed_buffer", {"ragged_buffer_factor": 1.0}),
      ("token_chunks", {"ragged_buffer_factor": 1.0, "num_moe_token_chunks": 2}),
      ("prefuse_token_chunks", {"num_moe_token_chunks": 2, "prefuse_moe_weights": True}),
  )
  def test_layer_matches_unfused(self, overrides):
    cfg_old = _cfg(**overrides)
    cfg_new = _cfg(**overrides, moe_router_grad_from_wo_input=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_old), cfg_old.mesh_axes)
    batch = int(cfg_old.per_device_batch_size * _NUM_DEVICES)
    x = jax.random.normal(jax.random.PRNGKey(7), (batch, cfg_old.max_target_length, cfg_old.emb_dim), jnp.float32)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_old.logical_axis_rules):
      _, params, _ = nnx.split(_build(cfg_old, mesh), nnx.Param, ...)
    (loss_old, out_old, grads_old), loops_old = _run_layer(cfg_old, mesh, params, x)
    (loss_new, out_new, grads_new), loops_new = _run_layer(cfg_new, mesh, params, x)

    np.testing.assert_array_equal(np.asarray(out_new, np.float32), np.asarray(out_old, np.float32))
    np.testing.assert_array_equal(np.asarray(loss_new, np.float32), np.asarray(loss_old, np.float32))
    leaves_old = jax.tree_util.tree_flatten_with_path(grads_old)[0]
    leaves_new = jax.tree_util.tree_leaves(grads_new)
    self.assertLen(leaves_new, len(leaves_old))
    for (path, a), b in zip(leaves_old, leaves_new):
      a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
      name = jax.tree_util.keystr(path)
      self.assertTrue(np.isfinite(b).all(), name)
      self.assertGreater(np.linalg.norm(a), 0.0, name)
      rel = _rel(b, a)
      print(f"{name}: rel_l2 = {rel:.3e}", file=sys.stderr)
      self.assertLess(rel, _GRAD_REL_L2, f"{name}: rel_l2 = {rel:.3e}")
    print(f"compiled remat-backward GEMM loops off/on = {loops_old}/{loops_new}", file=sys.stderr)
    # One GEMM loop fewer per chunk in the rematerialized backward: the wo GEMM is not replayed.
    self.assertEqual(
        loops_old - loops_new,
        cfg_new.num_moe_token_chunks,
        f"compiled remat-backward GEMM loops off/on = {loops_old}/{loops_new}",
    )


# ------------------------------------------------------------------------------ the fan-out without its row mask
def _unmasked_matmul(h, w, group_sizes, expert_assignments, group_offset):
  """A down projection whose weight gradient reads every row (all rows use expert 0): the positive control."""
  del group_sizes, expert_assignments, group_offset
  return (h.astype(jnp.float32) @ w[0].astype(jnp.float32)).astype(h.dtype)


@jax.custom_vjp
def _kernel_like_matmul(h, w, group_sizes, expert_assignments, group_offset):
  """_grouped_matmul with the worst case of the gmm_v2 kernels' VJP: the input gradient is computed for every row
  (a row outside the groups gives whatever its cotangent row gives, e.g. NaN) and the weight gradient masks the rows
  outside the groups on both operands, like the tgmm kernel."""
  return _grouped_matmul(h, w, group_sizes, expert_assignments, group_offset)


def _kernel_like_fwd(h, w, group_sizes, expert_assignments, group_offset):
  return _grouped_matmul(h, w, group_sizes, expert_assignments, group_offset), (h, w, group_sizes, group_offset)


def _kernel_like_bwd(res, g):
  h, w, group_sizes, group_offset = res
  local = _local_expert(h, group_sizes, group_offset)
  g32 = g.astype(jnp.float32)
  dh = jnp.einsum("rd,rfd->rf", g32, w[jnp.clip(local, 0, w.shape[0] - 1)].astype(jnp.float32))
  dw = jnp.stack([
      jnp.where((local == e)[:, None], h, 0).astype(jnp.float32).T @ jnp.where((local == e)[:, None], g32, 0.0)
      for e in range(w.shape[0])
  ])
  return dh.astype(h.dtype), dw.astype(w.dtype), None, None, None


_kernel_like_matmul.defvjp(_kernel_like_fwd, _kernel_like_bwd)


def _nan_outside_rows_fan_out(original):
  """_ring_unsort_fan_out that, when asked not to mask, leaves NaN in the rows outside the mask (as the SparseCore
  gather leaves them unwritten). Only the unmasked fan-out is touched; the forward sort and masked fan-outs are not."""

  def fan_out(*args, **kwargs):
    gathered, weight, mask, to_slot_order = original(*args, **kwargs)
    if not kwargs.get("mask_rows", True):
      gathered = jnp.where(mask[:, None], gathered, jnp.nan)
    return gathered, weight, mask, to_slot_order

  return fan_out


class FanOutRowMaskTest(parameterized.TestCase):
  """mask_fan_out_rows=False: rows outside the shard's range reach the down projection's VJP unmasked."""

  def setUp(self):
    super().setUp()
    original = ragged_sort._ring_unsort_fan_out  # pylint: disable=protected-access
    ragged_sort._ring_unsort_fan_out = _nan_outside_rows_fan_out(original)  # pylint: disable=protected-access
    self.addCleanup(setattr, ragged_sort, "_ring_unsort_fan_out", original)

  def test_no_mask_with_group_bounded_projection(self):
    # The grouped matmul's VJP reads no row outside the groups into dh or dwo: the result is finite and matches the
    # two ops apart, which mask the fanned-out rows. Full buffer only: the packed dropping buffer has no rows
    # outside the mask, so the injection would test nothing there.
    base = _op_run(None, fused=False)
    new = _op_run(None, fused=True, mask_fan_out_rows=False)
    np.testing.assert_array_equal(new[0], base[0], err_msg="forward output is not bit-identical")
    for name, a, b in zip(("dx", "dw", "dwi", "dwo"), base[1:], new[1:]):
      self.assertTrue(np.isfinite(b).all(), name)
      rel = _rel(b, a)
      self.assertLess(rel, _GRAD_REL_L2, f"{name}: rel_l2 vs two ops = {rel:.3e}")

  def test_no_mask_with_kernel_like_projection(self):
    # The input gradient's rows outside the groups are NaN here; the [rows, mlp] masks of dh and the routing-weight
    # gradient keep them out (full buffer: the rows of the other shards are outside the mask).
    base = _op_run(None, fused=False)
    new = _op_run(None, fused=True, mask_fan_out_rows=False, down_projection=_kernel_like_matmul)
    np.testing.assert_array_equal(new[0], base[0], err_msg="forward output is not bit-identical")
    for name, a, b in zip(("dx", "dw", "dwi", "dwo"), base[1:], new[1:]):
      self.assertTrue(np.isfinite(b).all(), name)
      rel = _rel(b, a)
      self.assertLess(rel, _GRAD_REL_L2, f"{name}: rel_l2 vs two ops = {rel:.3e}")

  def test_injected_rows_reach_the_vjp(self):
    # Positive control (full buffer: the rows of the other shards are outside the mask): with a projection whose
    # weight gradient reads every row, the injected NaN rows poison dwo without the fan-out mask and not with it.
    unmasked = _op_run(None, fused=True, mask_fan_out_rows=False, down_projection=_unmasked_matmul)
    masked = _op_run(None, fused=True, mask_fan_out_rows=True, down_projection=_unmasked_matmul)
    self.assertFalse(np.isfinite(unmasked[4]).all(), "dwo stayed finite: the injection did not reach the VJP")
    self.assertTrue(np.isfinite(masked[4]).all(), "dwo has NaN although the fan-out mask is on")


# --------------------------------------------------------------------------------------------------- the validator
class ConfigTest(parameterized.TestCase):
  """The key is rejected for each requirement it does not meet."""

  @parameterized.named_parameters(
      ("dense", {"sparse_matmul": False}, "sparse_matmul=True"),
      ("no_ring", {"use_ring_of_experts": False}, "use_ring_of_experts=True"),
      ("no_ragged_sort", {"use_ragged_sort": False}, "use_ragged_sort=True"),
      (
          "tensor_parallel",
          {"ici_expert_parallelism": 4, "ici_fsdp_parallelism": 1, "ici_tensor_parallelism": 2},
          "no tensor parallelism",
      ),
      ("mlp_bias", {"mlp_bias": True}, "mlp_bias=False"),
      # hy4-tiny sets router_matmul_precision=highest, which another validator rejects with quantization.
      ("quantization", {"quantization": "int8", "router_matmul_precision": ""}, "no quantization"),
      ("emb_chunks", {"num_moe_emb_chunks": 2, "use_gmm_v2": True}, "num_moe_emb_chunks=0"),
      # hy4-tiny's IndexCache (indexer_share_interval > 1) and router_matmul_precision are rejected with batch split
      # by other checks first.
      (
          "batch_split",
          {"use_batch_split_schedule": True, "indexer_share_interval": 1, "router_matmul_precision": ""},
          "use_batch_split_schedule=False",
      ),
      ("minimal_remat", {"remat_policy": "minimal"}, "remat_policy=minimal saves it as mlpwo"),
      ("save_dot_remat", {"remat_policy": "save_dot_except_mlpwi"}, "remat_policy=save_dot_except_mlpwi saves it"),
      ("saved_moe_mlpwo", {"remat_policy": "custom", "moe_mlpwo": "device"}, "saves it as moe_mlpwo"),
      ("offloaded_mlpwo", {"remat_policy": "custom", "mlpwo": "offload"}, "saves it as mlpwo"),
  )
  def test_rejected(self, overrides, requirement):
    with self.assertRaisesRegex(ValueError, f"moe_router_grad_from_wo_input=True requires: .*{re.escape(requirement)}"):
      _cfg(moe_router_grad_from_wo_input=True, **overrides)

  def test_accepted(self):
    self.assertTrue(_cfg(moe_router_grad_from_wo_input=True).moe_router_grad_from_wo_input)
    # A custom policy that saves other MoE tensors is fine.
    cfg = _cfg(moe_router_grad_from_wo_input=True, remat_policy="custom", moe_mlpwi_0="device", moe_x_sorted="device")
    self.assertTrue(cfg.moe_router_grad_from_wo_input)
    self.assertFalse(_cfg().moe_router_grad_from_wo_input)
