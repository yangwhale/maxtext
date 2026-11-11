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

"""DeepSeek Manifold-Constrained Hyper Connections (mHC) Layer."""

import functools
import itertools
import math
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config
from maxtext.common.common_types import HyperConnectionType
from maxtext.kernels import ihc as ihc_kernel
from maxtext.kernels.mhc import api as mhc_kernel
from maxtext.layers import nnx_wrappers
from maxtext.utils import accurate_math
from maxtext.layers.initializers import default_bias_init, default_scalar_init, nd_dense_init, variable_to_logically_partitioned
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils.sharding import get_logical_axis_rules, logical_to_mesh_axes


@functools.lru_cache(maxsize=None)
def get_permutation_matrices(k: int) -> Array:
  """Generates all permutation matrices of size k.

  Reference: mHC-lite: https://openreview.net/pdf?id=5IJX6kvOif
  Shape: (k!, k, k)
  """
  perms = list(itertools.permutations(range(k)))
  perms_array = jnp.array(perms)
  return jnp.eye(k)[perms_array]


def streams_flat(config) -> bool:
  """Whether the decoder carries hyper-connection streams flat, (batch, seq, k * emb), instead of 4D (batch, seq, k, emb).

  Only with the iHC Pallas kernel (its per-device view is (tokens, k * emb), so a flat carry makes that view a free
  reshape; a 4D carry is tiled over (k, emb) on TPU and copied into and out of the view at every layer). Layers that
  read the stream's trailing axes keep the 4D carry: engram (normalizes and unpacks the 4D stream) and an active
  dropout (broadcast over the stream axis, -2, which on a flat carry would be the sequence axis).
  """
  return (
      getattr(config, "mhc_expansion_rate", 1) > 1
      and getattr(config, "use_ihc_pallas_kernel", False)
      and not getattr(config, "engram_layers", None)
      and not (getattr(config, "enable_dropout", False) and getattr(config, "dropout_rate", 0.0) > 0.0)
  )


def get_functions(expansion_rate: int):
  """Creates functions to broadcast a single feature stream into multiple

  parallel paths (expand) and aggregate them back (reduce).
  """

  def expand(x: Array):
    # (batch, length, dim) -> (batch, length, streams, dim)
    return jnp.repeat(jnp.expand_dims(x, axis=2), expansion_rate, axis=2).astype(x.dtype)

  def reduce(x: Array):
    # (batch, length, streams, dim) -> (batch, length, dim)
    return jnp.sum(x, axis=2, dtype=x.dtype)

  return expand, reduce


def sinkhorn(t, iters=20):
  """Computes the Sinkhorn normalization of a matrix (rows and columns sum to 1)."""
  # Use float32 precision for numerical stability during normalization
  initial_dtype = t.dtype
  t = t.astype(jnp.float32)
  eps = 1e-6

  t = jax.nn.softmax(t, axis=-1) + eps
  t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)

  for _ in range(iters - 1):
    t = t / (jnp.sum(t, axis=-1, keepdims=True) + eps)
    t = t / (jnp.sum(t, axis=-2, keepdims=True) + eps)

  return t.astype(initial_dtype)


class ManifoldConstrainedHyperConnections(nnx.Module):
  """Implements Manifold-Constrained Hyper-Connections (mHC).

  Reference: https://arxiv.org/pdf/2512.24880

  Args:
      config: Configuration object containing hyperparameters.
      dim: The feature dimensionality.
      mesh: The hardware mesh for sharding.
      rngs: Random number generation in NNX.
  """

  def __init__(
      self,
      config: Config,
      dim: int,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sinkhorn_iterations = config.sinkhorn_iterations
    self.k = config.mhc_expansion_rate
    self.dim = dim
    self.rngs = rngs
    self.mesh = mesh
    self.dtype = self.config.dtype
    self.weight_dtype = self.config.weight_dtype
    self.matmul_precision = jax.lax.Precision(self.config.matmul_precision)

    if getattr(self.config, "use_mhc_pallas_kernel", False) and not self.config.enable_mhc_lite:
      raise ValueError("use_mhc_pallas_kernel=True requires enable_mhc_lite=True.")

    # Norm layer
    self.mhc_norm = RMSNorm(
        num_features=self.k * self.dim,
        dtype=self.config.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    # Scalars
    self.res_alpha_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.pre_alpha_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.post_alpha_scale = nnx.Param(
        default_scalar_init(self.rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )

    if self.config.enable_mhc_lite:
      num_perms = math.factorial(self.k)
      res_out_dim = num_perms
      res_beta_shape = (num_perms,)
      res_beta_sharding = (None,)
    else:
      res_out_dim = self.k * self.k
      res_beta_shape = (self.k, self.k)
      res_beta_sharding = (None, None)

    # Weight matrices
    scale_init = nd_dense_init(1.0, "fan_in", "normal")
    in_axis = 0
    out_axis = 1
    weight_sharding_axis_name = ("activation_embed", None)
    self.res_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, res_out_dim),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )
    self.pre_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )
    self.post_alpha = nnx.Param(
        scale_init(
            self.rngs.params(),
            (self.k * self.dim, self.k),
            self.weight_dtype,
            in_axis=in_axis,
            out_axis=out_axis,
        ),
        out_sharding=weight_sharding_axis_name,
    )

    # Biases
    self.res_beta = nnx.Param(
        default_bias_init(self.rngs.params(), res_beta_shape, self.weight_dtype),
        out_sharding=res_beta_sharding,
    )
    self.pre_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.post_beta = nnx.Param(
        default_bias_init(self.rngs.params(), (self.k,), self.weight_dtype),
        out_sharding=(None,),
    )

  def _get_mhc_weights(self) -> mhc_kernel.MhcWeights:
    """Collects layer parameters into a structured MhcWeights PyTree."""
    return mhc_kernel.MhcWeights(
        norm_scale=jnp.asarray(self.mhc_norm.scale[...], self.dtype),
        pre_alpha=jnp.asarray(self.pre_alpha[...], self.dtype),
        pre_bias=jnp.asarray(self.pre_beta[...], self.dtype),
        pre_scale=jnp.asarray(self.pre_alpha_scale[...], self.dtype),
        post_alpha=jnp.asarray(self.post_alpha[...], self.dtype),
        post_bias=jnp.asarray(self.post_beta[...], self.dtype),
        post_scale=jnp.asarray(self.post_alpha_scale[...], self.dtype),
        res_alpha=jnp.asarray(self.res_alpha[...], self.dtype),
        res_bias=jnp.asarray(self.res_beta[...], self.dtype),
        res_scale=jnp.asarray(self.res_alpha_scale[...], self.dtype),
    )

  def res_mapping(self, h_res: Array):
    """Helper function for residual mapping after matmul."""
    # In MaxText, we match weight precision to activations before Matmul
    res_beta = jnp.asarray(self.res_beta[...], self.dtype)
    res_alpha_scale = jnp.asarray(self.res_alpha_scale[...], self.dtype)

    if self.config.enable_mhc_lite:
      intermediate = res_alpha_scale * h_res + res_beta[None, None, :]
      # Use float32 for numerical stability during softmax
      weights = jax.nn.softmax(intermediate.astype(jnp.float32), axis=-1).astype(self.dtype)
      # Sum the permutation matrices with the weights
      permutation_matrices = get_permutation_matrices(self.k).astype(self.dtype)
      output = jnp.einsum(
          "bsn,nkm -> bskm",
          weights,
          permutation_matrices,
          precision=self.matmul_precision,
      )
      return output
    else:
      b, s, _ = h_res.shape
      h_res = jnp.reshape(h_res, (b, s, self.k, self.k))
      intermediate = res_alpha_scale * h_res + res_beta[None, None, :, :]
      output = sinkhorn(intermediate, self.sinkhorn_iterations)
      return output

  def mapping(self, h: Array, alpha_scale: Array, beta: Array, scale: float, eps: float = 0.0):
    """Helper function for both pre and post mappings after matmul."""
    # In MaxText, we match weight precision to activations before Matmul
    beta = jnp.asarray(beta, self.dtype)
    alpha_scale = jnp.asarray(alpha_scale, self.dtype)
    intermediate = alpha_scale * h + beta[None, None, :]
    output = scale * jax.nn.sigmoid(intermediate) + eps
    return output

  def __call__(
      self,
      norm_fn: Callable,
      branch_fn: Callable,
      x: Array,
      mhc_type: HyperConnectionType,
      **kwargs,
  ) -> Array:
    """Applying manifold-constrained hyper connection based on callable function.

    Args:
        norm_fn: The pre-normalization function to be applied.
        branch_fn: The function to be wrapped by the hyper-connection.
        x: Input tensor of shape `(batch..., dim)`.
        mhc_type: The variant of the connection to apply.
        **kwargs: Additional context passed to the branch function.

    Returns:
        The processed tensor, maintaining the shape of `x`.
    """
    # x shape: [batch, seq, expansion_rate, emb]
    b, s, k, d = x.shape

    h_post = None
    h_res = None
    context = None
    use_kernel = self.config.enable_mhc_lite and getattr(self.config, "use_mhc_pallas_kernel", False)
    if use_kernel:
      fwd_block_size = getattr(self.config, "mhc_pallas_kernel_fwd_block_size", 256)
      bwd_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_block_size", 128)
      bwd_feature_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_feature_block_size", 1024)
      kernel_config = mhc_kernel.MhcKernelConfig(
          block_size=fwd_block_size,
          bwd_block_size=bwd_block_size,
          bwd_feature_block_size=bwd_feature_block_size,
          rms_epsilon=self.config.normalization_layer_epsilon,
      )
      weights = self._get_mhc_weights()
      layer_input, context = mhc_kernel.pre(
          x,
          weights,
          jnp.asarray(get_permutation_matrices(self.k), self.dtype),
          config=kernel_config,
      )
    else:
      with jax.named_scope("mhc_norm"):
        # 1. Flatten the tensor, and RMS normalization
        norm_x = self.mhc_norm(jnp.reshape(x, (b, s, k * d)))

      # Fused Projections
      pre_alpha = jnp.asarray(self.pre_alpha[...], self.dtype)
      post_alpha = jnp.asarray(self.post_alpha[...], self.dtype)
      res_alpha = jnp.asarray(self.res_alpha[...], self.dtype)

      alpha_concat = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1)

      # MatMul on normalized input
      h_concat = jnp.einsum("bsm,mn -> bsn", norm_x, alpha_concat, precision=self.matmul_precision)
      h_pre = h_concat[..., : self.k]
      h_post = h_concat[..., self.k : 2 * self.k]
      h_res = h_concat[..., 2 * self.k :]

      # 2. Pre mapping
      pre_mapping = self.mapping(
          h_pre,
          self.pre_alpha_scale[...],
          self.pre_beta[...],
          1.0,
          eps=1e-6,
      )
      # bskd, bsk -> bsd (fused contracted GEMM)
      layer_input = jnp.einsum(
          "bsk,bskd->bsd",
          pre_mapping,
          x,
          precision=self.matmul_precision,
      )

    # 3. Pre-norm
    layer_input = norm_fn(layer_input)

    # 4. Attention or MLP
    metadata = {}
    if mhc_type == HyperConnectionType.ATTENTION:
      attention_outputs = branch_fn(inputs_q=layer_input, inputs_kv=layer_input, **kwargs)
      layer_out = attention_outputs[0]
      if len(attention_outputs) > 2:  # (out, kv_cache, indexer_mask) when the caller asked for the DSA mask
        metadata["indexer_mask"] = attention_outputs[2]
    elif mhc_type == HyperConnectionType.MLP_DENSE:
      layer_out = branch_fn(inputs=layer_input, **kwargs)
    elif mhc_type == HyperConnectionType.MLP_MOE:
      layer_out, load_balance_loss, moe_bias_updates = branch_fn(inputs=layer_input, **kwargs)
      metadata["load_balance_loss"] = load_balance_loss
      metadata["moe_bias_updates"] = moe_bias_updates
    else:
      raise ValueError(f"Unsupported type: {mhc_type}")

    if use_kernel:
      fwd_block_size = getattr(self.config, "mhc_pallas_kernel_fwd_block_size", 256)
      bwd_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_block_size", 128)
      bwd_feature_block_size = getattr(self.config, "mhc_pallas_kernel_bwd_feature_block_size", 1024)
      kernel_config = mhc_kernel.MhcKernelConfig(
          block_size=fwd_block_size,
          bwd_block_size=bwd_block_size,
          bwd_feature_block_size=bwd_feature_block_size,
      )
      output = mhc_kernel.post(
          layer_out,
          context,
          config=kernel_config,
      )
      return output, metadata

    # 5. Post mapping
    post_mapping = self.mapping(
        h_post,
        self.post_alpha_scale[...],
        self.post_beta[...],
        2.0,
    )
    # Moving away from einsum seems to allow XLA to perform better fusions
    # bsd,bsk -> bskd
    post_out = jnp.expand_dims(layer_out, axis=2) * jnp.expand_dims(post_mapping, axis=3)

    # 6. Residual mapping, res_out shape as [batch, seq, expansion_rate, emb]
    res_mapping = self.res_mapping(h_res)

    # bskm,bskd -> bsmd (fused contracted GEMM)
    res_out = jnp.einsum(
        "bskm,bskd->bsmd",
        res_mapping,
        x,
        precision=self.matmul_precision,
    )
    return res_out + post_out, metadata


class IdentityHyperConnections(nnx.Module):
  """Identity hyper-connections (iHC), the Hy4 variant of hyper-connections.

  Compared to mHC there is no residual mixing matrix (the residual of every stream is the identity,
  i.e. the sinkhorn part is dropped), the input RMS norm carries no learned scale, the post gate is
  `hc_post_magnitude * sigmoid(.) + hc_eps`, and gates plus the weighted stream sum are computed in
  float32. Per token, with the k streams flattened into `flat` of size k * dim:

    mixes = (flat @ [pre_alpha, post_alpha]) * rsqrt(mean(flat**2) + eps)
    pre   = sigmoid(pre_alpha_scale * mixes[:k] + pre_beta) + hc_eps
    post  = hc_post_magnitude * sigmoid(post_alpha_scale * mixes[k:] + post_beta) + hc_eps
    out_n = x_n + post_n * branch(norm(sum_n pre_n * x_n))

  Same call interface as `ManifoldConstrainedHyperConnections`.
  """

  def __init__(
      self,
      config: Config,
      dim: int,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.k = config.mhc_expansion_rate
    self.dim = dim
    self.mesh = mesh
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.eps = config.normalization_layer_epsilon
    self.hc_eps = config.hc_eps
    self.post_magnitude = config.hc_post_magnitude
    self.precision = jax.lax.Precision(config.hc_matmul_precision or "highest")

    weight_init = nnx.initializers.normal(stddev=config.hc_init_std)
    weight_sharding = ("activation_embed", None)
    self.pre_alpha = nnx.Param(
        weight_init(rngs.params(), (self.k * dim, self.k), self.weight_dtype), out_sharding=weight_sharding
    )
    self.post_alpha = nnx.Param(
        weight_init(rngs.params(), (self.k * dim, self.k), self.weight_dtype), out_sharding=weight_sharding
    )
    # pre_beta starts at -log(k - 1) so that the initial pre gates sum to ~1 over the k streams.
    pre_beta_init = -math.log(self.k - 1) if self.k > 1 else 0.0
    self.pre_beta = nnx.Param(jnp.full((self.k,), pre_beta_init, self.weight_dtype), out_sharding=(None,))
    self.post_beta = nnx.Param(jnp.zeros((self.k,), self.weight_dtype), out_sharding=(None,))
    self.pre_alpha_scale = nnx.Param(jnp.full((1,), 0.01, self.weight_dtype), out_sharding=(None,))
    self.post_alpha_scale = nnx.Param(jnp.full((1,), 0.01, self.weight_dtype), out_sharding=(None,))

  def _gate_params(self):
    """float32 gate parameters: alpha = [pre_alpha, post_alpha] (k*dim, 2k), then pre / post scale and beta."""
    alpha = jnp.concatenate(
        [jnp.asarray(self.pre_alpha[...], jnp.float32), jnp.asarray(self.post_alpha[...], jnp.float32)], axis=-1
    )
    pre_scale = jnp.asarray(self.pre_alpha_scale[...], jnp.float32)
    post_scale = jnp.asarray(self.post_alpha_scale[...], jnp.float32)
    pre_beta = jnp.asarray(self.pre_beta[...], jnp.float32)
    post_beta = jnp.asarray(self.post_beta[...], jnp.float32)
    return alpha, pre_scale, post_scale, pre_beta, post_beta

  def gates(self, x: Array) -> tuple[Array, Array]:
    """Returns float32 (pre, post) gates of shape [batch, seq, k]."""
    b, s, k, d = x.shape
    flat = jnp.reshape(x, (b, s, k * d)).astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + self.eps)
    alpha, pre_scale, post_scale, pre_beta, post_beta = self._gate_params()
    mixes = jnp.einsum("bsm,mn->bsn", flat, alpha, precision=self.precision) * inv_rms
    pre_logits, post_logits = mixes[..., :k], mixes[..., k:]
    acc = accurate_math.site_on(self.config, "sigmoid_ihc")  # accurate FP32 sigmoid (accurate_math)
    pre = accurate_math.sigmoid(pre_logits * pre_scale + pre_beta, acc) + self.hc_eps
    post = self.post_magnitude * accurate_math.sigmoid(post_logits * post_scale + post_beta, acc) + self.hc_eps
    return pre, post

  def _kernel_config(self) -> ihc_kernel.IhcKernelConfig:
    """Kernel config; the kernels run per device on the token shards of x (batch and sequence axes)."""
    mesh, token_axes = None, (None, None)
    if self.mesh is not None:
      logical = ("activation_batch", "activation_norm_length", "activation_embed")
      spec = logical_to_mesh_axes(logical, mesh=self.mesh, rules=get_logical_axis_rules())
      if spec[2] is not None:
        raise ValueError(f"use_ihc_pallas_kernel requires an unsharded embedding axis; activation_embed -> {spec[2]}.")
      mesh, token_axes = self.mesh, (spec[0], spec[1])
    return ihc_kernel.IhcKernelConfig(
        rms_epsilon=self.eps,
        hc_eps=self.hc_eps,
        post_magnitude=self.post_magnitude,
        block_size=self.config.ihc_pallas_block_size,
        bwd_block_size=self.config.ihc_pallas_bwd_block_size,
        mesh=mesh,
        token_axes=token_axes,
        interpret=jax.default_backend() != "tpu",
    )

  def _kernel_pre(self, kernel_config: ihc_kernel.IhcKernelConfig, x: Array):
    """Same gates and stream sum as `gates` + the float32 stream sum below, as one Pallas kernel."""
    k = self.k
    alpha, pre_scale, post_scale, pre_beta, post_beta = self._gate_params()
    scale = jnp.concatenate([jnp.broadcast_to(pre_scale, (k,)), jnp.broadcast_to(post_scale, (k,))])
    beta = jnp.concatenate([pre_beta, post_beta])
    return ihc_kernel.ihc_pre(kernel_config, x, alpha, scale, beta)

  def _kernel_fits(self, kernel_config: ihc_kernel.IhcKernelConfig, x: Array) -> bool:
    """Whether the per-device token count of x tiles into the kernel blocks (else the XLA path runs, e.g. decode)."""
    b, s = x.shape[0], x.shape[1]
    if kernel_config.mesh is not None:
      for i, entry in enumerate(kernel_config.token_axes):
        names = () if entry is None else (entry if isinstance(entry, tuple) else (entry,))
        shards = math.prod(kernel_config.mesh.shape[n] for n in names)
        if (b, s)[i] % shards:
          return False
        if i == 0:
          b //= shards
        else:
          s //= shards
    return ihc_kernel.tokens_supported(b * s, kernel_config)

  def __call__(
      self,
      norm_fn: Callable,
      branch_fn: Callable,
      x: Array,
      mhc_type: HyperConnectionType,
      **kwargs,
  ) -> Array:
    flat = x.ndim == 3  # streams carried flat, (batch, seq, k * dim); see streams_flat
    if flat:
      if x.shape[-1] != self.k * self.dim:
        raise ValueError(
            f"Expected hyper-connection streams, (batch, seq, k * dim) = (..., {self.k * self.dim}) or 4D; got {x.shape}."
        )
      x = jnp.reshape(x, x.shape[:2] + (self.k, self.dim))
    kernel_config = self._kernel_config() if self.config.use_ihc_pallas_kernel else None
    if kernel_config is not None and self._kernel_fits(kernel_config, x):
      layer_input, x_kernel, post = self._kernel_pre(kernel_config, x)
      layer_out, metadata = self._branch(norm_fn, branch_fn, layer_input, mhc_type, **kwargs)
      out = ihc_kernel.ihc_post(kernel_config, layer_out, x_kernel, post)
    else:
      out, metadata = self._xla_call(norm_fn, branch_fn, x, mhc_type, **kwargs)
    return (jnp.reshape(out, out.shape[:2] + (-1,)) if flat else out), metadata

  def _xla_call(self, norm_fn: Callable, branch_fn: Callable, x: Array, mhc_type: HyperConnectionType, **kwargs):
    """XLA path on 4D streams x (batch, seq, k, dim)."""
    pre, post = self.gates(x)
    xf = x.astype(jnp.float32)
    # Weighted stream sum, kept in float32. The plain form, jnp.sum(pre[..., None] * xf, axis=2), fails to compile with
    # the tpu_pre_training 0.2.5 image (jax 0.11.1, libtpu 0.0.46) on TPU v7x: the multiply-reduce fusion over
    # [batch, seq, k, emb] in the rematted backward hits "Cannot find a valid window for %multiply_reduce_fusion".
    # Two forms avoid it:
    #   - default: a contraction over k (einsum at HIGHEST, a multi-pass bfloat16 approximation of float32 on TPU);
    #   - ihc_stream_sum_unrolled: k explicit float32 multiply-adds (elementwise; fuses with the surrounding casts).
    #     Its backward still reduces over emb once per stream; it compiled and trained on a 64-chip v7x Hy4 run with
    #     remat (41 layers).
    # TODO: go back to jnp.sum (and drop ihc_stream_sum_unrolled) once the libtpu in use compiles it; re-test on v7x.
    if self.config.ihc_stream_sum_unrolled:
      layer_input = pre[..., 0:1] * xf[:, :, 0]
      for i in range(1, x.shape[2]):
        layer_input = layer_input + pre[..., i : i + 1] * xf[:, :, i]
      layer_input = self._round(layer_input).astype(x.dtype)
    else:
      layer_input = self._round(jnp.einsum("bsk,bskd->bsd", pre, xf, precision=jax.lax.Precision.HIGHEST)).astype(x.dtype)
    layer_out, metadata = self._branch(norm_fn, branch_fn, layer_input, mhc_type, **kwargs)
    out = xf + post[..., None] * jnp.expand_dims(layer_out.astype(jnp.float32), axis=2)
    return self._round(out).astype(x.dtype), metadata

  def _round(self, v: Array) -> Array:
    """ihc_force_bf16_rounding: round to `dtype` with reduce_precision, which XLA never removes (a plain
    float32 -> dtype -> float32 round trip may be elided under excess precision)."""
    if not getattr(self.config, "ihc_force_bf16_rounding", False) or jnp.dtype(self.dtype) == jnp.float32:
      return v
    fi = jnp.finfo(self.dtype)
    return jax.lax.reduce_precision(v, fi.nexp, fi.nmant)

  def _branch(self, norm_fn: Callable, branch_fn: Callable, layer_input: Array, mhc_type: HyperConnectionType, **kwargs):
    """Pre-norm and the wrapped branch; returns (layer_out, metadata)."""
    layer_input = norm_fn(layer_input)

    metadata = {}
    if mhc_type == HyperConnectionType.ATTENTION:
      attention_outputs = branch_fn(inputs_q=layer_input, inputs_kv=layer_input, **kwargs)
      layer_out = attention_outputs[0]
      if len(attention_outputs) > 2:  # (out, kv_cache, indexer_mask) when the caller asked for the DSA mask
        metadata["indexer_mask"] = attention_outputs[2]
    elif mhc_type == HyperConnectionType.MLP_DENSE:
      layer_out = branch_fn(inputs=layer_input, **kwargs)
    elif mhc_type == HyperConnectionType.MLP_MOE:
      layer_out, load_balance_loss, moe_bias_updates = branch_fn(inputs=layer_input, **kwargs)
      metadata["load_balance_loss"] = load_balance_loss
      metadata["moe_bias_updates"] = moe_bias_updates
    else:
      raise ValueError(f"Unsupported type: {mhc_type}")
    return layer_out, metadata


def make_hyper_connections(config: Config, dim: int, mesh: Mesh, rngs: nnx.Rngs):
  """Builds the hyper-connection module selected by `config.hc_variant`."""
  if config.hc_variant == "ihc":
    return IdentityHyperConnections(config, dim, mesh, rngs)
  return ManifoldConstrainedHyperConnections(config, dim, mesh, rngs)


class DeepSeek4HyperHead(nnx.Module):
  """Final HC-stream collapse; used by DeepSeek V4 before the shared RMSNorm."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.mhc_expansion_rate = config.mhc_expansion_rate
    self.emb_dim = config.emb_dim
    # DeepSeek-V4 uses 1e-6 for both (the config defaults); Hy4 sets hc_head_norm_epsilon to its rms_norm_eps.
    self.norm_eps = config.hc_head_norm_epsilon
    self.eps = config.hc_eps
    self.precision = jax.lax.Precision(config.hc_matmul_precision or config.matmul_precision)

    # Weight matrices
    weight_init = nd_dense_init(1.0, "fan_in", "normal")
    self.hc_fn = nnx.Param(
        weight_init(
            rngs.params(),
            (self.mhc_expansion_rate * self.emb_dim, self.mhc_expansion_rate),
            self.weight_dtype,
            in_axis=0,
            out_axis=1,
        ),
        out_sharding=("activation_embed", None),
    )
    self.hc_base = nnx.Param(
        default_bias_init(rngs.params(), (self.mhc_expansion_rate,), self.weight_dtype)
        + jnp.asarray(config.hc_head_base_init, self.weight_dtype),
        out_sharding=(None,),
    )
    self.hc_scale = nnx.Param(
        default_scalar_init(rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )

  def __call__(self, x: Array) -> Array:
    # x shape: [batch, length, k, d]
    b, s, k, d = x.shape
    assert k == self.mhc_expansion_rate
    assert d == self.emb_dim

    flat = jnp.reshape(x, (b, s, k * d))
    flat_f32 = flat.astype(jnp.float32)
    variance = jnp.mean(jnp.square(flat_f32), axis=-1, keepdims=True)
    flat_norm = flat_f32 * jax.lax.rsqrt(variance + self.norm_eps)

    hc_fn = jnp.asarray(self.hc_fn[...], jnp.float32)
    hc_base = jnp.asarray(self.hc_base[...], jnp.float32)
    hc_scale = jnp.asarray(self.hc_scale[...], jnp.float32)

    mixes = jnp.einsum("bsm,mk->bsk", flat_norm, hc_fn, precision=self.precision)
    acc = accurate_math.site_on(self.config, "sigmoid_hc_head")
    pre = accurate_math.sigmoid(mixes * hc_scale[None, None, :] + hc_base[None, None, :], acc) + self.eps

    x_f32 = x.astype(jnp.float32)
    out = jnp.sum(pre[:, :, :, None] * x_f32, axis=2)
    return out.astype(self.dtype)


DeepSeek4HyperHeadToLinen = nnx_wrappers.to_linen_class(
    DeepSeek4HyperHead,
    base_metadata_fn=variable_to_logically_partitioned,
)
