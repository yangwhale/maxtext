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

"""RMSNorm scale_grad_gpu_order: dgamma summed per sequence in float32, rounded to bf16 once, summed in float32."""

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx

from maxtext.layers.normalizations import RMSNorm

BF16 = ml_dtypes.bfloat16
B, S, D = 4, 64, 256


def _norm(**kw):
  return RMSNorm(D, epsilon=1e-5, dtype=jnp.bfloat16, weight_dtype=jnp.float32, single_rounding=True,
                 rngs=nnx.Rngs(params=0), **kw)


def _inputs(b=B):
  rng = np.random.default_rng(0)
  x = (rng.standard_normal((b, S, D)) * 3).astype(BF16)
  dy = (rng.standard_normal((b, S, D)) * 1e-2).astype(BF16)
  scale = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
  return x, dy, scale


def _grads(kw, x, dy, scale, jit=False):
  n = _norm(**kw)
  n.scale.value = jnp.asarray(scale)

  def loss(m, v):
    return jnp.sum(m(v).astype(jnp.float32) * jnp.asarray(dy, jnp.float32))

  f = nnx.jit(nnx.value_and_grad(loss, argnums=(0, 1))) if jit else nnx.value_and_grad(loss, argnums=(0, 1))
  val, (gm, gx) = f(n, jnp.asarray(x))
  y = n(jnp.asarray(x))
  return np.asarray(y.astype(jnp.float32)), np.asarray(gm.scale.value), np.asarray(gx.astype(jnp.float32)), float(val)


def _want_dgamma(x, dy):
  """FP64 per-sequence sums of normed * dy, each rounded to bf16, summed (TE dgamma + Megatron main_grad order)."""
  x64 = x.astype(np.float64)
  normed = x64 / np.sqrt(np.mean(x64 * x64, -1, keepdims=True) + 1e-5)
  per_row = np.sum(normed * dy.astype(np.float64), axis=1)
  return per_row.astype(np.float32).astype(BF16).astype(np.float64).sum(0)


class RmsNormScaleGradGpuOrderTest(unittest.TestCase):

  def test_forward_and_input_grad_unchanged(self):
    x, dy, scale = _inputs()
    y0, _, gx0, v0 = _grads({"scale_grad_fp32": True}, x, dy, scale)
    y1, _, gx1, v1 = _grads({"scale_grad_gpu_order": True}, x, dy, scale)
    np.testing.assert_array_equal(y0, y1)
    self.assertEqual(v0, v1)
    np.testing.assert_allclose(gx0, gx1, rtol=0, atol=0)

  def test_input_grad_bitwise_same_as_default_and_fp32(self):
    x, dy, scale = _inputs()
    for jit in (False, True):
      y0, _, gx0, _ = _grads({}, x, dy, scale, jit=jit)
      _, _, gx1, _ = _grads({"scale_grad_fp32": True}, x, dy, scale, jit=jit)
      y2, _, gx2, _ = _grads({"scale_grad_gpu_order": True}, x, dy, scale, jit=jit)
      np.testing.assert_array_equal(y0, y2)
      np.testing.assert_array_equal(gx0, gx2)
      np.testing.assert_array_equal(gx1, gx2)

  def test_upstream_mixing_grads_unchanged(self):
    """hc_head-like use: streams [b, s, k, d] mixed by learnable weights, then RMSNorm. The mixing weights' gradient
    is a near-cancelling sum (the norm's input gradient is orthogonal to its input), so it is very sensitive to how
    the input gradient is computed; it must be bitwise the same with the switch on and off."""
    rng = np.random.default_rng(1)
    k = 4
    xs = jnp.asarray((rng.standard_normal((B, S, k, D)) * 2).astype(np.float32)).astype(jnp.bfloat16)
    dy = jnp.asarray((rng.standard_normal((B, S, D)) * 1e-2).astype(np.float32)).astype(jnp.bfloat16)
    w = jnp.asarray((0.25 + 0.05 * rng.standard_normal(k)).astype(np.float32))
    scale = jnp.asarray((1 + 0.1 * rng.standard_normal(D)).astype(np.float32))

    def grad_w(kw, jit):
      n = _norm(**kw)
      n.scale.value = scale

      def loss(w):
        out = jnp.sum(w[None, None, :, None] * xs.astype(jnp.float32), axis=2).astype(jnp.bfloat16)
        return jnp.sum(n(out).astype(jnp.float32) * dy.astype(jnp.float32))

      f = jax.jit(jax.grad(loss)) if jit else jax.grad(loss)
      return np.asarray(f(w))

    for jit in (False, True):
      g0 = grad_w({}, jit)
      np.testing.assert_array_equal(g0, grad_w({"scale_grad_gpu_order": True}, jit))
      np.testing.assert_array_equal(g0, grad_w({"scale_grad_fp32": True}, jit))

  def test_streams_shape_dgamma(self):
    """4-D input [b, s, k, d]: dgamma is summed per batch row over s and k."""
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((2, 16, 4, D)) * 3).astype(BF16)
    dy = (rng.standard_normal((2, 16, 4, D)) * 1e-2).astype(BF16)
    scale = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    _, g, _, _ = _grads({"scale_grad_gpu_order": True}, x, dy, scale, jit=True)
    x64 = x.astype(np.float64)
    normed = x64 / np.sqrt(np.mean(x64 * x64, -1, keepdims=True) + 1e-5)
    per_row = np.sum(normed * dy.astype(np.float64), axis=(1, 2))
    want = per_row.astype(np.float32).astype(BF16).astype(np.float64).sum(0)
    self.assertGreaterEqual(np.mean(np.abs(g - want) <= np.abs(want) * 2.0**-8), 0.999)

  def test_dgamma_per_sequence_bf16_then_fp32_sum(self):
    x, dy, scale = _inputs()
    want = _want_dgamma(x, dy)
    for jit in (False, True):
      _, g, _, _ = _grads({"scale_grad_gpu_order": True}, x, dy, scale, jit=jit)
      self.assertEqual(g.dtype, np.float32)
      # each row is one bf16 rounding of an FP32 sum; FP32 summation order can move a row by one bf16 ulp only on a
      # rounding boundary, so compare against the exactly rounded FP64 order with a tolerance of that size.
      close = np.abs(g - want) <= np.abs(want) * 2.0**-8
      self.assertGreaterEqual(np.mean(close), 0.999, jit)
      self.assertGreaterEqual(np.mean(g == want.astype(np.float32)), 0.9, jit)

  def test_one_sequence_lands_on_bf16_grid_under_jit(self):
    # reduce_precision must survive XLA's excess-precision simplifications
    x, dy, scale = _inputs(b=1)
    _, g, _, _ = _grads({"scale_grad_gpu_order": True}, x, dy, scale, jit=True)
    np.testing.assert_array_equal(g, g.astype(BF16).astype(np.float32))
    _, g32, _, _ = _grads({"scale_grad_fp32": True}, x, dy, scale, jit=True)
    self.assertLess(np.mean(g32 == g32.astype(BF16).astype(np.float32)), 0.5)

  def test_differs_from_both_existing_orders(self):
    x, dy, scale = _inputs()
    _, g_gpu, _, _ = _grads({"scale_grad_gpu_order": True}, x, dy, scale)
    _, g_fp32, _, _ = _grads({"scale_grad_fp32": True}, x, dy, scale)
    _, g_def, _, _ = _grads({}, x, dy, scale)
    # default: the whole (all-row) sum rounded once to bf16
    np.testing.assert_array_equal(g_def, g_def.astype(BF16).astype(np.float32))
    self.assertFalse(np.array_equal(g_gpu, g_fp32))
    self.assertFalse(np.array_equal(g_gpu, g_def))


if __name__ == "__main__":
  unittest.main()
