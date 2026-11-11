# Copyright 2023–2026 Google LLC
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

"""Utilities for QK-Clip (Muon Clip)."""

import jax
import jax.numpy as jnp
from flax import nnx


def _get_key_name(k):
  """Helper to unwrap JAX path keys."""
  if hasattr(k, "key"):
    return k.key
  if hasattr(k, "idx"):
    return k.idx
  return k


def calculate_max_logit_metric(intermediate_outputs):
  """Extracts and computes the global maximum logit from intermediate outputs.

  Recognizes two shapes: Linen sow stores `(array,)` so the leaf path ends in
  `max_logits, 0`; NNX `nnx.Intermediate(array)` stores the array directly so
  the leaf path ends in `max_logits`.

  Returns the global max scalar, or None if no logits were found.
  """
  all_max_logits = []

  def extract_logits(path, val):
    if not path:
      return
    last_key = _get_key_name(path[-1])
    parent_key = _get_key_name(path[-2]) if len(path) >= 2 else None
    if last_key == "max_logits" or parent_key == "max_logits":
      all_max_logits.append(val)

  jax.tree_util.tree_map_with_path(extract_logits, intermediate_outputs)

  if not all_max_logits:
    return None

  return jnp.max(jnp.stack([jnp.max(x) for x in all_max_logits]))


def _check_attention_type(config):
  if getattr(config, "attention_type", None) != "mla":
    raise ValueError(
        f"QK-Clip is only supported for MLA attention (attention_type='mla'). "
        f"Current configuration: {getattr(config, 'attention_type', 'None')}"
    )


def _max_logits_at(curr):
  """Read max_logits from a node in the intermediates tree.

  Returns the [batch, num_heads] array, or None if not present. Handles both
  the Linen sow shape (`{"max_logits": (array,)}`) and the NNX shape
  (`{"max_logits": array}` or `{"attention_op": {"max_logits": array}}`).
  """
  if not isinstance(curr, dict):
    return None
  ml = curr.get("max_logits")
  if ml is None and "attention_op" in curr and isinstance(curr["attention_op"], dict):
    ml = curr["attention_op"].get("max_logits")
  if ml is None:
    return None
  if isinstance(ml, (tuple, list)):
    return ml[0] if ml else None
  return ml


def _clip_scale(s_max, tau, denominator_eps):
  """min(1, tau / (s_max + eps)); eps 1e-6 by default, 0 with qk_clip_mcore_denominator (Megatron-Core).

  A head whose max logit is not positive is below tau (> 0) and keeps scale 1: tau / s_max would be negative (or -0
  for -inf) and its square root in _clip_mla_weight NaN. Real weights have such heads, all their weight on the sink
  (visible with qk_clip_max_logit_exclude_sink). Megatron-Core only clips when some head exceeds tau and would hit the
  same NaN then; for every positive s_max the scale is unchanged."""
  return jnp.where(s_max > 0, jnp.minimum(1.0, tau / (s_max + denominator_eps)), 1.0)


def _scale_from_max_logits(max_logits_batch, tau, denominator_eps=1e-6):
  """One scale per head from an unscanned layer's max_logits [..., num_heads]."""
  axes = tuple(range(max_logits_batch.ndim - 1))
  s_max = jnp.max(max_logits_batch, axis=axes)
  return _clip_scale(s_max, tau, denominator_eps)


def _kernel_scale(max_logits_batch, param, tau, param_scan_axis, denominator_eps=1e-6):
  """QK-Clip scale for one wq_b / wkv_b kernel, shaped to broadcast against it: one value per layer and head.

  Unscanned kernel [rank, heads, dim], max_logits [batch, heads]: one scale per head -> [heads, 1].
  Scanned kernel with the layer axis at `param_scan_axis` ([rank, layers, heads, dim] by default); the scan stacks
  its max_logits on axis 0 ([layers, batch, heads]). The max is taken per layer, so each layer is clipped by its own
  max logit, as QK-Clip is defined; reducing over the layer axis as well would clip every layer of the scan by the
  largest logit among them.
  """
  if param.ndim == 4:
    num_layers, num_heads = param.shape[param_scan_axis], param.shape[-2]
    if not 0 <= param_scan_axis < param.ndim - 2:
      raise ValueError(f"QK-Clip: param_scan_axis={param_scan_axis} must come before the head axis of {param.shape}.")
    if max_logits_batch.ndim < 2 or max_logits_batch.shape[0] != num_layers or max_logits_batch.shape[-1] != num_heads:
      raise ValueError(
          f"QK-Clip: scanned kernel {param.shape} (layers on axis {param_scan_axis}) does not match max_logits"
          f" {max_logits_batch.shape}; expected [layers, ..., heads]."
      )
    s_max = jnp.max(max_logits_batch, axis=tuple(range(1, max_logits_batch.ndim - 1)))  # [layers, heads]
    scale = _clip_scale(s_max, tau, denominator_eps)
    shape = [1] * param.ndim
    shape[param_scan_axis], shape[-2] = num_layers, num_heads
    return scale.reshape(shape)
  return jnp.expand_dims(_scale_from_max_logits(max_logits_batch, tau, denominator_eps), axis=-1)


def _clip_mla_weight(layer_name, param, scale_b, qk_nope):
  """Apply a per-(layer,) head scale, already shaped to broadcast against the kernel, to a wq_b or wkv_b kernel."""
  head = param[..., :qk_nope]
  tail = param[..., qk_nope:]
  head_new = head * jnp.sqrt(scale_b)
  if layer_name == "wq_b":
    tail_new = tail * scale_b
  else:  # wkv_b: tail is the V slice, untouched
    tail_new = tail
  return jnp.concatenate([head_new, tail_new], axis=-1)


def apply_qk_clip(state, intermediate_outputs, config):
  """Applies QK-Clip to MLA weights based on max_logits (Linen path).

  Returns a new TrainState with `wq_b`/`wkv_b` kernels rescaled per-head.
  """
  _check_attention_type(config)
  tau = float(config.qk_clip_threshold)

  def clip_mla_weights(path, param):
    if len(path) < 2:
      return param
    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param

    curr = intermediate_outputs.get("intermediates", intermediate_outputs)
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param

    max_logits_batch = _max_logits_at(curr)
    if max_logits_batch is None:
      return param

    scale_b = _kernel_scale(
        max_logits_batch,
        param,
        tau,
        getattr(config, "param_scan_axis", 1),
        0.0 if getattr(config, "qk_clip_mcore_denominator", False) else 1e-6,
    )
    return _clip_mla_weight(layer_name, param, scale_b, config.qk_nope_head_dim)

  new_params = jax.tree_util.tree_map_with_path(clip_mla_weights, state.params)
  return state.replace(params=new_params)


def apply_qk_clip_nnx(state, intermediate_outputs, config):
  """Applies QK-Clip to MLA weights on an NNX TrainStateNNX.

  `state.model` is mutated in place (NNX modules are mutable). Returns `state`
  so call sites can use the same `new_state = apply_qk_clip(...)` pattern as
  the Linen path.

  The intermediates tree mirrors the NNX module hierarchy, so `max_logits`
  sowed by `AttentionOp` lives at `...self_attention.attention_op.max_logits`.
  We accept either that shape or `...self_attention.max_logits` (matching the
  Linen-side fixtures and small-test setups).
  """
  _check_attention_type(config)
  tau = float(config.qk_clip_threshold)

  _, params_state, _ = nnx.split(state.model, nnx.Param, ...)
  params_dict = params_state.to_pure_dict()  # pyrefly: ignore[missing-attribute]

  def clip_mla_weights(path, param):
    if len(path) < 2:
      return param
    layer_name = _get_key_name(path[-2])
    if layer_name not in ("wq_b", "wkv_b"):
      return param

    curr = intermediate_outputs
    for node in path[:-2]:
      key = _get_key_name(node)
      if isinstance(curr, dict) and key in curr:
        curr = curr[key]
      else:
        return param

    max_logits_batch = _max_logits_at(curr)
    if max_logits_batch is None:
      return param

    scale_b = _kernel_scale(
        max_logits_batch,
        param,
        tau,
        getattr(config, "param_scan_axis", 1),
        0.0 if getattr(config, "qk_clip_mcore_denominator", False) else 1e-6,
    )
    return _clip_mla_weight(layer_name, param, scale_b, config.qk_nope_head_dim)

  new_params_dict = jax.tree_util.tree_map_with_path(clip_mla_weights, params_dict)
  nnx.replace_by_pure_dict(params_state, new_params_dict)
  nnx.update(state.model, params_state)
  return state


def per_head_stats(params_dict, intermediate_outputs, config):
  """Per-(layer,) head max logit and QK-Clip scale of every attention module that QK-Clip touches (qk_clip_log_per_head).

  Walks the same paths as apply_qk_clip_nnx (each wq_b kernel and the max_logits sown beside it) and reduces the
  max_logits the same way as _kernel_scale. Returns {"<module path>": [2, (layers,) heads]} with row 0 the max logit
  and row 1 the scale min(1, tau / (max logit + eps)). Logging only.
  """
  tau = float(config.qk_clip_threshold)
  eps = 0.0 if getattr(config, "qk_clip_mcore_denominator", False) else 1e-6
  scan_axis = getattr(config, "param_scan_axis", 1)
  out = {}

  def visit(path, param):
    if len(path) < 2 or _get_key_name(path[-2]) != "wq_b":
      return
    curr = intermediate_outputs
    for node in path[:-2]:
      key = _get_key_name(node)
      if not (isinstance(curr, dict) and key in curr):
        return
      curr = curr[key]
    ml = _max_logits_at(curr)
    if ml is None:
      return
    if param.ndim == 4:
      s_max = jnp.max(ml, axis=tuple(range(1, ml.ndim - 1)))  # [layers, heads]
      if s_max.shape != (param.shape[scan_axis], param.shape[-2]):
        raise ValueError(f"QK-Clip: max_logits {ml.shape} do not match the scanned kernel {param.shape}.")
    else:
      s_max = jnp.max(ml, axis=tuple(range(ml.ndim - 1)))  # [heads]
    out["/".join(_get_key_name(k) for k in path[:-2])] = jnp.stack([s_max, _clip_scale(s_max, tau, eps)])

  jax.tree_util.tree_map_with_path(visit, params_dict)
  return out
