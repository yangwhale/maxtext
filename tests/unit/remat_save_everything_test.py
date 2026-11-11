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

from maxtext.layers.nnx_decoders import NNXDecoder


def _policy(name):
  fake = types.SimpleNamespace(config=types.SimpleNamespace(remat_policy=name))
  return NNXDecoder.get_remat_policy(fake)


class RematSaveEverythingTest(unittest.TestCase):

  def test_policy_is_everything_saveable(self):
    self.assertIs(_policy("save_everything"), jax.checkpoint_policies.everything_saveable)

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


if __name__ == "__main__":
  unittest.main()
