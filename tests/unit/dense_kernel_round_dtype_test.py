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


"""DenseGeneral kernel_round_dtype (indexer_weights_proj_bf16_weight): BF16 weight values, FP32 dot, straight-through grad."""

import unittest

import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.layers.linears import DenseGeneral


def _dense(round_dtype):
  return DenseGeneral(
      in_features_shape=16, out_features_shape=4, dtype=jnp.float32, weight_dtype=jnp.float32,
      matmul_precision="highest", kernel_round_dtype=round_dtype, rngs=nnx.Rngs(0),
  )


class KernelRoundDtypeTest(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    self.w = (rng.standard_normal((16, 4)) * 0.3).astype(np.float32)
    self.x = jnp.asarray(rng.standard_normal((2, 3, 16)).astype(np.float32)).astype(jnp.bfloat16).astype(jnp.float32)

  def _out_and_grad(self, round_dtype):
    m = _dense(round_dtype)
    m.kernel.value = jnp.asarray(self.w)
    grads = nnx.grad(lambda mod: jnp.sum(mod(self.x) ** 2))(m)
    return np.asarray(m(self.x), np.float64), np.asarray(grads.kernel.value, np.float64)

  def test_off_uses_fp32_master(self):
    out, _ = self._out_and_grad(None)
    np.testing.assert_allclose(out, np.asarray(self.x, np.float64) @ self.w.astype(np.float64), rtol=1e-5, atol=1e-6)

  def test_on_uses_bf16_values_with_fp32_output(self):
    out, g = self._out_and_grad(jnp.bfloat16)
    wb = np.asarray(jnp.asarray(self.w).astype(jnp.bfloat16).astype(jnp.float32), np.float64)
    ref = np.asarray(self.x, np.float64) @ wb
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    self.assertGreater(np.abs(out - np.asarray(self.x, np.float64) @ self.w.astype(np.float64)).max(), 1e-6)
    # straight-through: d/dW sum(y^2) = x^T (2 y), y computed with the rounded weight; the gradient itself is not rounded
    x2 = np.asarray(self.x, np.float64).reshape(-1, 16)
    g_ref = x2.T @ (2 * ref.reshape(-1, 4))
    np.testing.assert_allclose(g, g_ref, rtol=1e-4, atol=1e-5)
    self.assertFalse(np.array_equal(g, np.asarray(jnp.asarray(g, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32))))


if __name__ == "__main__":
  unittest.main()
