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

"""splash_dsinks_gpu_order: the attention-sink gradient in Megatron-Core's order (GPU, hy4_mcore with attn_scores_fp32).

GPU (autograd through the float32 softmax over [keys, sink]): dp_j = do . v_j is a bf16 matmul output (rounded to bf16
per position), p = softmax in float32, and the sink column's gradient is p_sink * (0 - sum_j p_j * dp_j), summed over
queries in float32:  d_sink = -sum_q p_sink[q] * sum_j p_j[q] * bf16(dp_j[q]).

The tokamax splash kernel never forms dp per position in its sink path: its own d_sinks rounds p_sink, o and do terms to
bf16 (tokamax 0.0.14), and splash_dsinks_fp32 uses di = sum_d o . do (o in bf16) instead of sum_j p_j dp_j. The dkv / dq
kernels do form dp per block, but in float32 and not exposed. So this recomputes the sum outside the kernel, in plain
JAX, chunked over queries: scores q . k in float32 (HIGHEST: exact products of the bf16 inputs), the row's log-sum-exp
over the allowed keys and the sink, p and p_sink in float32, dp = bf16(do . v^T), all sums in float32.

with_gpu_order_sink_grad(out, sinks, q, k, v, allowed) is the identity on out in the forward; in the backward it passes
the output cotangent through and returns this d_sink for sinks (nothing for q, k, v: the caller feeds the kernel
stop_gradient(sinks), so the kernel's own d_sink is dropped and dq / dk / dv stay the kernel's). Costs one dense
attention-sized recompute per call in the backward.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


def gpu_order_dsinks(sinks, q, k, v, allowed, do, chunk: int = 256):  # pylint: disable=too-many-positional-arguments
  """d_sink [heads] for one sequence: q [h, Q, d], k [h, K, d], v [h, K, dv], allowed bool [Q, K], do [h, Q, dv]."""
  h, nq, _ = q.shape
  chunk = min(chunk, nq)
  pad = (-nq) % chunk
  if pad:
    q = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
    do = jnp.pad(do, ((0, 0), (0, pad), (0, 0)))
    allowed = jnp.pad(allowed, ((0, pad), (0, 0)))
  n = q.shape[1] // chunk
  qc = q.reshape(h, n, chunk, -1).transpose(1, 0, 2, 3)
  doc = do.reshape(h, n, chunk, -1).transpose(1, 0, 2, 3)
  mc = allowed.reshape(n, chunk, -1)
  s32 = sinks.astype(jnp.float32)
  hi = jax.lax.Precision.HIGHEST

  def one(xs):
    q_, do_, m_ = xs
    s = jnp.einsum("hqd,hkd->hqk", q_.astype(jnp.float32), k.astype(jnp.float32), precision=hi)
    s = jnp.where(m_[None], s, -jnp.inf)
    mx = jnp.maximum(jnp.max(s, axis=-1), s32[:, None])  # [h, c]
    denom = jnp.sum(jnp.exp(s - mx[..., None]), axis=-1) + jnp.exp(s32[:, None] - mx)
    lse = mx + jnp.log(denom)
    p = jnp.exp(s - lse[..., None])  # masked keys: exp(-inf) = 0
    p_sink = jnp.exp(s32[:, None] - lse)
    dp = jnp.einsum("hqd,hkd->hqk", do_, v, preferred_element_type=jnp.float32)
    # bf16 rounding of dp with reduce_precision: XLA may drop a float32 -> bf16 -> float32 round trip (excess precision)
    dp = jax.lax.reduce_precision(dp, exponent_bits=8, mantissa_bits=7)
    d_row = jnp.sum(p * dp, axis=-1)  # sum_j p_j dp_j, [h, c]
    return -jnp.sum(p_sink * d_row, axis=-1)  # [h]

  return jnp.sum(jax.lax.map(one, (qc, doc, mc)), axis=0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def with_gpu_order_sink_grad(out, sinks, q, k, v, allowed, chunk=256):  # pylint: disable=too-many-positional-arguments
  """Identity on `out`; its backward gives `sinks` the GPU-order d_sink (see the module docstring)."""
  del sinks, q, k, v, allowed, chunk
  return out


def _fwd(out, sinks, q, k, v, allowed, chunk):  # pylint: disable=too-many-positional-arguments
  del chunk
  return out, (sinks, q, k, v, allowed)


def _bwd(chunk, res, do):
  sinks, q, k, v, allowed = res
  ds = gpu_order_dsinks(sinks, q, k, v, allowed, do.astype(v.dtype), chunk)
  zero_int = np.zeros(allowed.shape, dtype=jax.dtypes.float0)
  return do, ds.astype(sinks.dtype), jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), zero_int


with_gpu_order_sink_grad.defvjp(_fwd, _bwd)
