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


"""ihc_force_bf16_rounding: the iHC branch input and updated streams are rounded to dtype with reduce_precision."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers.mhc import IdentityHyperConnections


def _stub(on, dtype=jnp.bfloat16):
  return types.SimpleNamespace(config=types.SimpleNamespace(ihc_force_bf16_rounding=on), dtype=dtype)


class IhcForceBf16RoundingTest(unittest.TestCase):

  def setUp(self):
    self.v = jnp.asarray(np.random.default_rng(0).standard_normal((2, 8, 16)), jnp.float32)

  def test_off_is_identity(self):
    out = IdentityHyperConnections._round(_stub(False), self.v)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(self.v))

  def test_on_rounds_to_bf16_values_and_survives_a_cast_round_trip(self):
    f = jax.jit(lambda v: IdentityHyperConnections._round(_stub(True), v).astype(jnp.bfloat16).astype(jnp.float32))
    want = np.asarray(self.v.astype(jnp.bfloat16).astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(f(self.v)), want)
    out = np.asarray(IdentityHyperConnections._round(_stub(True), self.v))
    np.testing.assert_array_equal(out, want)  # already bf16-representable while still float32

  def test_float32_dtype_is_untouched(self):
    out = IdentityHyperConnections._round(_stub(True, jnp.float32), self.v)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(self.v))


if __name__ == "__main__":
  unittest.main()
