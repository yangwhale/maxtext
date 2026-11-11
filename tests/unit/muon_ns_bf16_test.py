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

"""muon_mcore_ns_bf16: Newton-Schulz on BF16 tensors, as Megatron-Core's emerging_optimizers 0.3.0 at
fp32_matmul_prec="medium" (newton_schulz: X.to(bfloat16); A = X X^T; B = addmm(A, A, A, beta=b, alpha=c);
X = addmm(X, B, X, beta=a, alpha=1)). The reference below is that algorithm in numpy with ml_dtypes: BF16 operands,
FP32 accumulation and epilogue, one BF16 rounding of A, B and X per step."""

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from optax.contrib import MuonDimensionNumbers as mdn

from maxtext.optimizers.muon_mcore_compat import _orthogonalize, muon_compat

NS = (3.4445, -4.7750, 2.0315)


def _bf16(v):
  return np.asarray(v, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _reference(x):
  x = np.asarray(x, np.float32)
  t = x.shape[0] > x.shape[1]
  if t:
    x = x.T
  m = _bf16(x / max(np.linalg.norm(x.astype(np.float64)), 1e-7))
  a, b, c = NS
  for _ in range(5):
    A = _bf16(m @ m.T)
    B = _bf16(b * A + c * (A @ A))
    m = _bf16(a * m + B @ m)
  return m.T if t else m


def _rel(a, b):
  return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


class MuonNsBf16Test(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    # Low-rank-ish gradients, as in practice: the BF16 iterations then differ from FP32 by percents.
    u = rng.standard_normal((96, 4)) @ rng.standard_normal((4, 160))
    self.x = (u + 0.05 * rng.standard_normal((96, 160))).astype(np.float32) * 1e-3

  def _orth(self, x, bf16):
    return np.asarray(
        _orthogonalize(jnp.asarray(x), jnp.asarray(NS), 5, 1e-8, mdn((0,), (1,)), True, bf16), np.float64
    )

  def test_matches_bf16_reference(self):
    for x in (self.x, self.x.T):  # wide and tall (the tall one is transposed inside)
      ref = _reference(x)
      got = self._orth(x, True)
      # Only the summation order of the FP32 accumulation differs: about a quarter of the values land one BF16 ulp
      # apart (rel L2 about 2e-3); the FP32 iterations are about 7e-2 away (test_off_is_fp32_and_differs).
      self.assertLess(_rel(got, ref), 5e-3)
      # Every value is a BF16 number: the last rounding was not dropped.
      np.testing.assert_array_equal(got, _bf16(got))

  def test_off_is_fp32_and_differs(self):
    ref = _reference(self.x)
    off = self._orth(self.x, False)
    with jax.default_matmul_precision("highest"):
      off_hi = self._orth(self.x, False)
    on = self._orth(self.x, True)
    self.assertGreater(_rel(off_hi, ref), 5 * _rel(on, ref))  # negative control: FP32 iterations are far from BF16
    self.assertFalse(np.array_equal(off, _bf16(off)))

  def test_flag_reaches_the_optimizer(self):
    params = {"w": jnp.zeros((96, 160))}
    dn = {"w": mdn((0,), (1,))}
    kw = {"learning_rate": 1.0, "ns_coeffs": NS, "beta": 0.95, "eps": 1e-10, "muon_weight_dimension_numbers": dn,
          "mcore_nesterov": True, "mcore_ns_normalize": True}
    g = {"w": jnp.asarray(self.x)}
    outs = {}
    for flag in (False, True):
      tx = muon_compat(**kw, mcore_ns_bf16=flag)
      up, _ = tx.update(g, tx.init(params), params)
      outs[flag] = np.asarray(up["w"], np.float64)
    # Step 1: direction = (1 - beta^2) g, scale sqrt(max(1, 160 / 96)); update = -scale * NS(direction).
    want = -np.sqrt(160 / 96) * _reference((1 - 0.95**2) * self.x)
    self.assertLess(_rel(outs[True], want), 5e-3)
    self.assertGreater(_rel(outs[False], want), 5 * _rel(outs[True], want))


if __name__ == "__main__":
  unittest.main()
