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

"""remat_policy=save_everything: the backward pass recomputes no forward op."""

import types
import unittest

import jax
import jax.numpy as jnp

from maxtext.layers.nnx_decoders import NNXDecoder, save_everything_except_complex


def _policy(name):
  fake = types.SimpleNamespace(config=types.SimpleNamespace(remat_policy=name))
  return NNXDecoder.get_remat_policy(fake)


class RematSaveEverythingTest(unittest.TestCase):

  def test_policy(self):
    self.assertIs(_policy("save_everything"), save_everything_except_complex)

  def test_backward_has_no_recompute(self):
    def f(x):
      return jnp.sum(jnp.sin(jnp.tanh(x) * 3.0))

    x = jnp.linspace(-1.0, 1.0, 16)
    for name, expect_sin in (("save_everything", False), ("full", True)):
      g = jax.checkpoint(f, policy=_policy(name))
      jaxpr = str(jax.make_jaxpr(jax.grad(g))(x))
      # With everything saved, the backward only uses residuals: no second tanh / sin in the gradient program.
      self.assertEqual(jaxpr.count("tanh") > 1, expect_sin, name)
      self.assertTrue(jnp.allclose(jax.grad(g)(x), jax.grad(f)(x)))

  def test_complex_intermediates(self):
    """Complex RoPE (as in YaRN): everything_saveable fails (jax inserts reduce_precision on complex residuals);
    save_everything recomputes only the complex values and keeps the real ones saved."""

    def f(x, freqs):
      h = jnp.tanh(x) * 3.0
      first, second = jnp.split(h, 2, axis=-1)
      rotated = (first + 1j * second) * freqs
      return jnp.sum(jnp.sin(jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)))

    x = jnp.linspace(-1.0, 1.0, 16).reshape(2, 8)
    freqs = jnp.exp(1j * jnp.linspace(0.0, 1.0, 4))
    with self.assertRaisesRegex(TypeError, "reduce_precision does not accept dtype complex"):
      jax.grad(jax.checkpoint(f, policy=jax.checkpoint_policies.everything_saveable))(x, freqs)
    g = jax.checkpoint(f, policy=_policy("save_everything"))
    self.assertTrue(jnp.allclose(jax.grad(g)(x, freqs), jax.grad(f)(x, freqs)))
    self.assertEqual(str(jax.make_jaxpr(jax.grad(g))(x, freqs)).count("tanh"), 1)  # real intermediates still saved


if __name__ == "__main__":
  unittest.main()
