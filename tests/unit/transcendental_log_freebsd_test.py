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

"""transcendental_log_freebsd: float32 log through the FreeBSD e_logf.c port (accurate_math.log_freebsd).

- accuracy: at most 1 ULP against float64 on every float32 in [0.5, 4) and on log-uniform samples in [1e-37, 1e37]
  and [1, 4096) (the log-sum-exp domain), plus the smallest normal; zero, subnormals and other special values as jnp.log;
- gradients: the derivative is t / x, so cross entropy and log_softmax gradients equal those of the HIGHEST path
  bit for bit; only the forward value changes;
- log_mode: "freebsd" only when both switches are on.
"""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.utils import accurate_math as am
from maxtext.utils import max_utils


def _ulp(out, x):
  """|out - float64 log correctly rounded to float32| in float32 steps (ordered bit distance, signs handled)."""
  cr = np.log(x.astype(np.float64)).astype(np.float32)

  def key(a):
    i = np.ascontiguousarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)

  return np.abs(key(out) - key(cr))


class LogFreebsdTest(unittest.TestCase):

  def _check(self, x):
    out = np.asarray(jax.jit(am.log_freebsd)(jnp.asarray(x)))
    d = _ulp(out, x)
    self.assertLessEqual(int(d.max()), 1, f"max {int(d.max())} ULP at x = {x[int(d.argmax())]!r}")
    return d

  def test_all_float32_in_half_to_four(self):
    lo, hi = np.float32(0.5).view(np.uint32), np.float32(4.0).view(np.uint32)
    d = self._check(np.arange(lo, hi, dtype=np.uint32).view(np.float32))
    self.assertGreater(float((d == 0).mean()), 0.9)  # correctly rounded for most inputs, like CUDA logf

  def test_log_uniform(self):
    rng = np.random.default_rng(1234)
    self._check(np.exp(rng.uniform(np.log(1e-37), np.log(1e37), 1 << 20)).astype(np.float32))
    self._check(np.exp(rng.uniform(0.0, np.log(4096.0), 1 << 20)).astype(np.float32))

  def test_subnormal_and_specials(self):
    self._check(np.array([1.1754944e-38, 2e-38, 3.4e38], np.float32))  # smallest normal, near the top
    sp = np.array([0.0, -1.0, np.inf, np.nan, 1.0, 1e-45, 1e-40, 1.17e-38], np.float32)  # subnormals: as jnp.log
    out = np.asarray(am.log_freebsd(jnp.asarray(sp)))
    np.testing.assert_array_equal(out, np.asarray(jnp.log(jnp.asarray(sp))))

  def test_bf16_in_bf16_out(self):
    x = jnp.asarray([0.5, 1.0, 3.0], jnp.bfloat16)
    self.assertEqual(am.log_freebsd(x).dtype, jnp.bfloat16)

  def test_derivative_is_t_over_x(self):
    x = jnp.asarray(np.linspace(0.01, 50.0, 1001, dtype=np.float32))
    g = jax.grad(lambda v: jnp.sum(am.log_freebsd(v)))(x)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(jax.grad(lambda v: jnp.sum(jnp.log(v)))(x)))

  def test_log_mode(self):
    on = SimpleNamespace(transcendental_log_freebsd=True)
    off = SimpleNamespace(transcendental_log_freebsd=False)
    self.assertEqual(am.log_mode(on, True), "freebsd")
    self.assertIs(am.log_mode(on, False), False)
    self.assertIs(am.log_mode(off, True), True)
    self.assertIs(am.log_mode(SimpleNamespace(), True), True)

  def test_cross_entropy_gradients_unchanged(self):
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(0, 3, (4, 7, 512)).astype(np.float32))
    targets = jax.nn.one_hot(jnp.asarray(rng.integers(0, 512, (4, 7))), 512)

    def f(lg, mode):
      return jnp.sum(max_utils.cross_entropy_with_logits(lg, targets, 0.0, accurate=mode)[0])

    g1 = jax.grad(lambda v: f(v, True))(logits)
    g2 = jax.grad(lambda v: f(v, "freebsd"))(logits)
    np.testing.assert_array_equal(np.asarray(g1), np.asarray(g2))
    np.testing.assert_allclose(float(f(logits, "freebsd")), float(f(logits, True)), rtol=1e-6)

  def test_log_softmax_gradients_unchanged(self):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(0, 2, (3, 64)).astype(np.float32))
    w = jnp.asarray(rng.normal(0, 1, (3, 64)).astype(np.float32))
    g1 = jax.grad(lambda v: jnp.sum(w * am.log_softmax(v, -1, True)))(x)
    g2 = jax.grad(lambda v: jnp.sum(w * am.log_softmax(v, -1, "freebsd")))(x)
    np.testing.assert_array_equal(np.asarray(g1), np.asarray(g2))


if __name__ == "__main__":
  unittest.main()
