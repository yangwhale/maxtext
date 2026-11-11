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

"""Normalization Layers."""

import functools
from typing import Any

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import Array, DType, ShardMode, is_fp8_dtype
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import Initializer, variable_to_logically_partitioned
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import truncate_out_sharding


def _align_scale_with_normalized_axis(scale: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Reshards a 1-D norm scale to match the sharding of the axis it normalizes.

  The scale is stored with the `kernel_axes` layout, which is picked for the
  usual case of normalizing over the embedding axis (`norm` -> `tensor`). A
  per-head QK norm instead normalizes over `head_dim`, which is unsharded while
  the neighbouring `heads` axis is the one on `tensor`. Multiplying those
  directly under `ShardMode.EXPLICIT` would place `tensor` on two axes of the
  result, which JAX rejects, so align the scale with the activation instead.

  This is a no-op whenever the two already agree, which includes every
  auto-sharding-equivalent layout for the ordinary layer norms.
  """
  # Read through `.partitions`: a scale carrying a reduced/unreduced tag (the deferred
  # data-parallel all-reduce under gradient accumulation) rejects direct indexing. The tags
  # are carried over to the new spec so the scale keeps its place on the deferred path.
  activation_axis = jax.typeof(y).sharding.spec.partitions[-1]
  scale_spec = jax.typeof(scale).sharding.spec
  if scale_spec.partitions[-1] == activation_axis:
    return scale
  return jax.sharding.reshard(
      scale,
      jax.sharding.PartitionSpec(activation_axis, unreduced=scale_spec.unreduced, reduced=scale_spec.reduced),
  )


def _scale_mul(normed, scale_r, sharding):
  """The single_rounding path's product: exactly the op (an einsum, i.e. a dot_general) the default path uses."""
  return jnp.einsum("...k,k->...k", normed, scale_r, out_sharding=sharding)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _scale_mul_dgamma_per_row(normed: jnp.ndarray, scale: jnp.ndarray, dtype, sharding) -> jnp.ndarray:
  """normed * round_to_dtype(scale), float32 in and out, with the scale gradient summed like Megatron-Core + TE.

  TE's fused RMSNorm backward sums dgamma over all tokens of its micro-batch in float32 and writes it once in the
  parameter dtype (bf16); Megatron's DDP hook then adds it into the float32 main_grad, so micro-batches and data
  parallel ranks are summed in float32. With micro_batch_size 1 per rank, one rank's micro-batch is one sequence.
  Here: dgamma per batch row (= sequence; float32 sum over every other axis, across any sequence sharding),
  rounded once to `dtype` with reduce_precision (never removed as excess precision), then summed over rows in
  float32. The forward op and the input-gradient op are the ones autodiff produces for the default path (the
  same einsum and its transpose), so only the scale gradient changes.
  """
  fi = jnp.finfo(dtype)
  return _scale_mul(normed, lax.reduce_precision(scale, fi.nexp, fi.nmant), sharding)


def _scale_mul_dgamma_per_row_fwd(normed, scale, dtype, sharding):
  return _scale_mul_dgamma_per_row(normed, scale, dtype, sharding), (normed, scale)


def _scale_mul_dgamma_per_row_bwd(dtype, sharding, res, g):
  normed, scale = res
  fi = jnp.finfo(dtype)
  scale_r = lax.reduce_precision(scale, fi.nexp, fi.nmant)
  _, vjp_normed = jax.vjp(lambda n: _scale_mul(n, scale_r, sharding), normed)
  (d_normed,) = vjp_normed(g)
  if normed.ndim == 1:
    return d_normed, lax.reduce_precision(normed * g.astype(jnp.float32), fi.nexp, fi.nmant)
  axes = tuple(range(1, normed.ndim - 1))
  per_row = jnp.sum(normed * g.astype(jnp.float32), axis=axes) if axes else normed * g.astype(jnp.float32)
  per_row = lax.reduce_precision(per_row, fi.nexp, fi.nmant)  # [batch, features]: one bf16 rounding per sequence
  return d_normed, jnp.sum(per_row, axis=0).astype(scale.dtype)


_scale_mul_dgamma_per_row.defvjp(_scale_mul_dgamma_per_row_fwd, _scale_mul_dgamma_per_row_bwd)


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = jax.nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      with_scale: bool = True,
      single_rounding: bool = False,
      scale_grad_fp32: bool = False,
      scale_grad_gpu_order: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    # single_rounding: compute x * rsqrt(mean(x^2) + eps) * scale entirely in float32 and round to `dtype`
    # once at the end (as fused GPU RMSNorm kernels do), instead of rounding the normalized value to `dtype`
    # first and then multiplying by the `dtype` scale (two roundings). The scale is still rounded to `dtype`
    # before the multiply, matching a GPU model whose parameters are kept in `dtype`.
    self.single_rounding = single_rounding
    # scale_grad_fp32 (single_rounding only): round the scale to `dtype` with a straight-through gradient, so the
    # scale gradient (a sum over all tokens) stays float32 instead of being rounded to `dtype` by the transpose of
    # reduce_precision. The forward value is unchanged.
    self.scale_grad_fp32 = scale_grad_fp32
    # scale_grad_gpu_order (single_rounding only; wins over scale_grad_fp32): the scale gradient is summed over each
    # batch row (sequence) in float32, rounded once to `dtype`, then summed over rows in float32 -- the order of
    # TE's fused RMSNorm dgamma + Megatron's float32 main_grad with micro_batch_size 1 per rank. Forward unchanged.
    self.scale_grad_gpu_order = scale_grad_gpu_order
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = dtype if is_fp8_dtype(weight_dtype) else weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.with_scale = with_scale
    if self.with_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), self.weight_dtype),
          out_sharding=kernel_axes,
      )
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    if self.single_rounding and self.with_scale and self.scale_offset == 0.0:
      normed_fp32 = x * lax.rsqrt(mean2 + self.epsilon)
      scale = self.scale.get_value()
      if self.parameter_memory_host_offload:
        scale = jax.device_put(scale, max_utils.device_space())
      # Round the scale to `dtype` (as the GPU's `dtype` parameter) with reduce_precision: a convert round trip
      # may be optimized away under excess precision, reduce_precision never is.
      if self.scale_grad_gpu_order and self.shard_mode != ShardMode.EXPLICIT:
        y = _scale_mul_dgamma_per_row(normed_fp32, jnp.asarray(scale, jnp.float32), self.dtype, None)
        return jnp.asarray(y, self.dtype)
      if self.scale_grad_fp32:
        scale_fp32 = max_utils.round_to_dtype_straight_through(jnp.asarray(scale, jnp.float32), self.dtype)
      else:
        fi = jnp.finfo(self.dtype)
        scale_fp32 = lax.reduce_precision(jnp.asarray(scale, jnp.float32), fi.nexp, fi.nmant)
      if self.shard_mode == ShardMode.EXPLICIT:
        scale_fp32 = align_scale_with_normalized_axis(scale_fp32, normed_fp32)
      sharding = truncate_out_sharding(out_sharding, normed_fp32.ndim) if (
          self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None) else None
      y = jnp.einsum("...k,k->...k", normed_fp32, scale_fp32, out_sharding=sharding)
      return jnp.asarray(y, self.dtype)

    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)

    # out_sharding must be None in auto shard mode
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    if out_sharding is not None:
      out_sharding = truncate_out_sharding(out_sharding, y.ndim)

    if not self.with_scale:
      if out_sharding is not None:
        y = jax.lax.with_sharding_constraint(y, out_sharding)
      return y

    scale = self.scale.get_value()
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    scale_fp32 = jnp.asarray(scale, jnp.float32)
    effective_scale = scale_fp32 + self.scale_offset
    if self.shard_mode == ShardMode.EXPLICIT:
      effective_scale = _align_scale_with_normalized_axis(effective_scale, y)

    if self.scale_offset != 0.0:
      normed_fp32 = x * lax.rsqrt(mean2 + self.epsilon)
      y = jnp.einsum("...k,k->...k", normed_fp32, effective_scale, out_sharding=out_sharding)
      return jnp.asarray(y, self.dtype)

    effective_scale = jnp.asarray(effective_scale, self.dtype)
    return jnp.einsum("...k,k->...k", y, effective_scale, out_sharding=out_sharding)


class GlobalRMSNorm(RMSNorm):
  """
  Applies RMSNorm over the last two dimensions (Heads * HeadDim).
  Used for Olmo3 which normalizes across all heads combined.
  """

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    # x shape: [..., Heads, HeadDim]
    input_shape = x.shape

    # Flatten the last two dimensions: [..., Heads * HeadDim]
    # We use -2 and -1 to ensure we capture the last two dims regardless of rank
    flattened_shape = input_shape[:-2] + (input_shape[-2] * input_shape[-1],)
    x_flat = x.reshape(flattened_shape)

    # Apply standard RMSNorm (which normalizes over the last axis)
    y_flat = super().__call__(x_flat, out_sharding)

    # Reshape back to [..., Heads, HeadDim]
    return y_flat.reshape(input_shape)


def Qwen3NextRMSNorm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: DType = None,
    weight_dtype: DType = None,
    shard_mode=None,
    kernel_axes=None,
    parameter_memory_host_offload=None,
    *,
    rngs: nnx.Rngs,
):
  """Used for input and post attention layernorms in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1. The learnable scale parameter `scale` is initialized to ZEROS.
  2. The scale is applied as `(1.0 + self.scale)`, making the initial scale effectively 1.0.
     This matches the PyTorch implementation of Qwen3NextRMSNorm.
  """
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=epsilon,
          dtype=dtype,
          weight_dtype=weight_dtype,
          shard_mode=shard_mode if shard_mode is not None else ShardMode.AUTO,
          kernel_axes=kernel_axes if kernel_axes is not None else (),
          scale_init=jax.nn.initializers.zeros,
          scale_offset=1.0,
          parameter_memory_host_offload=bool(parameter_memory_host_offload),
          rngs=rngs,
      )
  )


class Qwen3NextRMSNormGated(nnx.Module):
  """
  This applies RMS Normalization and then a gated activation function (SiLU).
  This is used within the Qwen3NextGatedDeltaNet.

  The normalization is performed by an internal `RMSNorm` instance (`self.rms_norm`),
  which has its own learnable `scale` parameter, initialized to ONES.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
    weight_dtype: The datatype of the internal RMSNorm scale.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float,
      dtype: DType,
      weight_dtype: DType,
      shard_mode: ShardMode = ShardMode.AUTO,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=self.epsilon,
            dtype=jnp.float32,
            weight_dtype=weight_dtype,
            shard_mode=shard_mode,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(
      self,
      hidden_states: Array,
      gate: Array,
      out_sharding: NamedSharding | None = None,
  ) -> Array:
    """Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F) where F is
        num_features.
      out_sharding: Optional layout for the normalized states, honoured only
        under `ShardMode.EXPLICIT`.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    """
    normalized_states = self.rms_norm(hidden_states, out_sharding=out_sharding)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


def rms_norm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    shard_mode: ShardMode = ShardMode.AUTO,
    kernel_axes: tuple[None | str, ...] = (),
    scale_init: Initializer = jax.nn.initializers.ones,
    name: None | str = None,
    parameter_memory_host_offload: bool = False,
    with_scale: bool = True,
):
  """Creates a RMSNorm module."""
  module = nnx_wrappers.to_linen(
      RMSNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      shard_mode=shard_mode,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      with_scale=with_scale,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  """L2 normalization function. Normalizes a vector to have a length of 1.

  Args:
    x: Input array.
    dim: The axis or axes along which to normalize. Defaults to the last axis.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array with the same shape as x.
  """

  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
  return x * inv_norm


Qwen3NextRMSNormLinen = nnx_wrappers.to_linen_class(
    RMSNorm,
    base_metadata_fn=variable_to_logically_partitioned,
    scale_init=jax.nn.initializers.zeros,
    scale_offset=1.0,
)
