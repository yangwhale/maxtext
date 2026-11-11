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

"""GateLogit bf16_round_kernel: the router matmul sees the kernel rounded to bf16 (RNE), the activation untouched."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from maxtext.layers import moe


def _gate(bf16_round_kernel):
  mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
  return moe.GateLogit(
      in_features_shape=64,
      out_features_shape=16,
      model_name="hunyuan3-smoke",
      mesh=mesh,
      rngs=nnx.Rngs(params=0),
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      score_func="",
      matmul_precision="highest",
      bf16_round_kernel=bf16_round_kernel,
  )


class GateBf16RoundingTest(unittest.TestCase):

  def test_kernel_rounded_activation_untouched(self):
    rng = np.random.default_rng(0)
    kernel = rng.standard_normal((64, 16)).astype(np.float32)  # not bf16-representable
    x = rng.standard_normal((2, 5, 64)).astype(np.float32)  # FP32 activation: must not be rounded
    k_bf16 = np.asarray(jnp.asarray(kernel, jnp.bfloat16).astype(jnp.float32))
    self.assertGreater(np.abs(k_bf16 - kernel).max(), 0)
    for flag, k_used in ((False, kernel), (True, k_bf16)):
      gate = _gate(flag)
      gate.kernel.value = jnp.asarray(kernel)
      out, _ = gate(jnp.asarray(x))
      ref = np.einsum("bsd,de->bse", x.astype(np.float64), k_used.astype(np.float64))
      np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=1e-5, atol=1e-5)
      self.assertEqual(out.dtype, jnp.float32)

  def test_gradient_reaches_fp32_kernel(self):
    gate = _gate(True)
    x = np.random.default_rng(1).standard_normal((1, 3, 64)).astype(np.float32)

    def loss(g):
      return jnp.sum(g(jnp.asarray(x))[0])

    grads = nnx.grad(loss)(gate)
    self.assertEqual(grads.kernel.value.dtype, jnp.float32)
    # d loss / d kernel[d, e] = sum over tokens of x[..., d]: an FP32 value, not rounded to bf16 by the forward rounding
    want = np.broadcast_to(x.astype(np.float64).sum((0, 1))[:, None], (64, 16))
    np.testing.assert_allclose(np.asarray(grads.kernel.value, np.float64), want, rtol=1e-6, atol=1e-6)
    want_bf16 = np.asarray(jnp.asarray(want, jnp.bfloat16).astype(jnp.float32))
    self.assertGreater(np.abs(want_bf16 - want).max(), 1e-4)

  def test_rounding_survives_jit(self):
    rng = np.random.default_rng(2)
    kernel = rng.standard_normal((64, 16)).astype(np.float32)
    x = rng.standard_normal((2, 5, 64)).astype(np.float32)
    gate = _gate(True)
    gate.kernel.value = jnp.asarray(kernel)
    out = nnx.jit(lambda g, v: g(v)[0])(gate, jnp.asarray(x))
    k_bf16 = np.asarray(jnp.asarray(kernel, jnp.bfloat16).astype(jnp.float32)).astype(np.float64)
    ref = np.einsum("bsd,de->bse", x.astype(np.float64), k_bf16)
    np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
