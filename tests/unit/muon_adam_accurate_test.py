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

"""muon_adam_accurate: the plain AdamW branch of optax_muon with accurate float32 arithmetic. Same formula and state
as optax adamw (nesterov=False); the reference is that formula in float64 numpy."""

import unittest

import jax.numpy as jnp
import numpy as np
from optax.contrib import MuonDimensionNumbers as mdn

from maxtext.optimizers.muon_mcore_compat import _div, _one_minus_pow, _recip, _sqrt, muon_compat

B1, B2, EPS, WD, LR = 0.9, 0.95, 1e-8, 0.1, 1e-3


def _adam64(gs, p0):
  m = np.zeros_like(p0)
  v = np.zeros_like(p0)
  p = p0.copy()
  ups = []
  for t, g in enumerate(gs, 1):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    u = -LR * ((m / (1 - B1**t)) / (np.sqrt(v / (1 - B2**t)) + EPS) + WD * p)
    ups.append(u)
    p = p + u
  return ups


class MuonAdamAccurateTest(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    # Tiny gradients, as the iHC gate parameters see (around 1e-9 .. 1e-6).
    self.gs = [(rng.standard_normal(64) * 10.0 ** rng.uniform(-9, -6, 64)).astype(np.float32) for _ in range(6)]
    self.p0 = rng.standard_normal(64).astype(np.float32) * 0.1

  def _run(self, accurate):
    params = {"w": jnp.zeros((8, 16)), "b": jnp.asarray(self.p0)}
    dn = {"w": mdn((0,), (1,)), "b": None}
    tx = muon_compat(
        learning_rate=LR,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        beta=0.95,
        eps=EPS,
        adam_b1=B1,
        adam_b2=B2,
        adam_weight_decay=WD,
        muon_weight_dimension_numbers=dn,
        adam_nesterov=False,
        adam_accurate=accurate,
    )
    st = tx.init(params)
    out = []
    for g in self.gs:
      up, st = tx.update({"w": jnp.zeros((8, 16)), "b": jnp.asarray(g)}, st, params)
      out.append(np.asarray(up["b"], np.float64))
      params = {"w": params["w"], "b": params["b"] + up["b"]}
    return out

  def test_matches_fp64_adam(self):
    want = _adam64([g.astype(np.float64) for g in self.gs], self.p0.astype(np.float64))
    got, off = self._run(True), self._run(False)
    for w, a, o in zip(want, got, off):
      rel_a = np.linalg.norm(a - w) / np.linalg.norm(w)
      self.assertLess(rel_a, 1e-6)
      # On CPU optax's own adamw is accurate too: the two arms agree to float32 rounding.
      self.assertLess(np.linalg.norm(a - o) / np.linalg.norm(w), 1e-6)

  def test_recip_is_close_to_division(self):
    x = jnp.asarray(np.random.default_rng(1).uniform(1e-3, 1e3, 4096), jnp.float32)
    want = 1.0 / np.asarray(x, np.float64)
    ulp = np.spacing(want.astype(np.float32)).astype(np.float64)
    self.assertLessEqual(np.max(np.abs(np.asarray(_recip(x), np.float64) - want) / ulp), 1.5)

  def test_div(self):
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(4096) * np.exp(rng.uniform(-20, 20, 4096))).astype(np.float32)
    y = np.exp(rng.uniform(-20, 20, 4096)).astype(np.float32)
    want = x.astype(np.float64) / y.astype(np.float64)
    ulp = np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
    self.assertLessEqual(np.max(np.abs(np.asarray(_div(jnp.asarray(x), jnp.asarray(y)), np.float64) - want) / ulp), 1.5)

  def test_one_minus_pow(self):
    t = jnp.arange(1, 1001, dtype=jnp.int32)
    for b in (0.9, 0.95, 0.999):
      want = 1.0 - b ** np.arange(1, 1001, dtype=np.float64)
      got = np.asarray(_one_minus_pow(b, t), np.float64)
      ulp = np.spacing(want.astype(np.float32)).astype(np.float64)
      self.assertLessEqual(np.max(np.abs(got - want) / ulp), 4.0, b)

  def test_sqrt(self):
    x = np.concatenate([[0.0], np.exp(np.random.default_rng(2).uniform(-60, 60, 4096))]).astype(np.float32)
    got = np.asarray(_sqrt(jnp.asarray(x)), np.float64)
    want = np.sqrt(x.astype(np.float64))
    ulp = np.spacing(want.astype(np.float32)).astype(np.float64)
    self.assertEqual(got[0], 0.0)
    self.assertLessEqual(np.max(np.abs(got - want) / ulp), 1.0)

  def test_needs_plain_adam(self):
    with self.assertRaises(ValueError):
      muon_compat(
          learning_rate=LR,
          ns_coeffs=(3.4445, -4.7750, 2.0315),
          beta=0.95,
          eps=EPS,
          nesterov=True,
          muon_weight_dimension_numbers={"b": None},
          adam_accurate=True,
      )


if __name__ == "__main__":
  unittest.main()
