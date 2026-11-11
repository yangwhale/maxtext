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

"""RMSNorm single_rounding: normalize * scale in float32, rounded to bf16 once."""

import unittest

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx

from maxtext.layers.normalizations import RMSNorm

BF16 = ml_dtypes.bfloat16
D = 256


def _norm(single_rounding):
  return RMSNorm(D, epsilon=1e-5, dtype=jnp.bfloat16, weight_dtype=jnp.float32,
                 single_rounding=single_rounding, rngs=nnx.Rngs(params=0))


def _inputs():
  rng = np.random.default_rng(0)
  x = (rng.standard_normal((64, D)) * 3).astype(BF16)
  scale = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)  # FP32 master weight, not bf16-representable
  return x, scale


def _ref64(x, scale):
  """FP64 truth of x * rsqrt(mean(x^2) + eps) * bf16(scale)."""
  x64 = x.astype(np.float64)
  s64 = scale.astype(BF16).astype(np.float64)
  return x64 / np.sqrt(np.mean(x64 * x64, -1, keepdims=True) + 1e-5) * s64


class RmsNormSingleRoundingTest(unittest.TestCase):

  def _run(self, flag, jit=False):
    x, scale = _inputs()
    n = _norm(flag)
    n.scale.value = jnp.asarray(scale)
    y = nnx.jit(lambda m, v: m(v))(n, jnp.asarray(x)) if jit else n(jnp.asarray(x))
    self.assertEqual(y.dtype, jnp.bfloat16)
    return np.asarray(y.astype(jnp.float32)), x, scale

  def test_single_rounding_is_one_rounding_of_fp32_formula(self):
    y, x, scale = self._run(True)
    x32 = x.astype(np.float32)
    r32 = x32 / np.sqrt(np.mean(x32 * x32, -1, keepdims=True) + np.float32(1e-5)) * scale.astype(BF16).astype(np.float32)
    want = r32.astype(BF16).astype(np.float32)
    # FP32 rsqrt / reduction order may differ in the last FP32 bit; that can only move a bf16 result by one ulp
    # when the FP32 value sits on a rounding boundary.
    self.assertGreaterEqual(np.mean(y == want), 0.999)

  def test_more_correctly_rounded_than_default(self):
    want = _ref64(*_inputs()).astype(BF16).astype(np.float32)
    on, _, _ = self._run(True)
    off, _, _ = self._run(False)
    frac_on, frac_off = np.mean(on == want), np.mean(off == want)
    self.assertGreaterEqual(frac_on, 0.999)
    self.assertLess(frac_off, frac_on)

  def test_jit_keeps_single_rounding(self):
    want = _ref64(*_inputs()).astype(BF16).astype(np.float32)
    on, _, _ = self._run(True, jit=True)
    self.assertGreaterEqual(np.mean(on == want), 0.999)

  def test_default_unchanged(self):
    y, x, scale = self._run(False)
    x32 = x.astype(np.float32)
    normed = (x32 / np.sqrt(np.mean(x32 * x32, -1, keepdims=True) + np.float32(1e-5))).astype(BF16)
    want = (normed.astype(np.float32) * scale.astype(BF16).astype(np.float32)).astype(BF16).astype(np.float32)
    self.assertGreaterEqual(np.mean(y == want), 0.999)


if __name__ == "__main__":
  unittest.main()
