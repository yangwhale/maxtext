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

"""DenseGeneral fp32_accumulation: the dot accumulates in float32 and is rounded to the compute dtype once."""

import unittest

import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.layers import linears


def _dense(flag):
  return linears.DenseGeneral(
      in_features_shape=256,
      out_features_shape=32,
      dtype=jnp.bfloat16,
      weight_dtype=jnp.float32,
      rngs=nnx.Rngs(params=0),
      fp32_accumulation=flag,
  )


def _emulated_windowed(x, k, chunks):
  """Partial products over contraction chunks, each rounded to bf16 and summed in bf16 (what a windowed matmul does)."""
  acc = None
  for xs, ks in zip(np.split(x, chunks, axis=-1), np.split(k, chunks, axis=0)):
    p = (xs.astype(np.float64) @ ks.astype(np.float64)).astype(np.float32)
    p = np.asarray(jnp.asarray(p, jnp.bfloat16).astype(jnp.float32))
    acc = p if acc is None else np.asarray(jnp.asarray(acc + p, jnp.bfloat16).astype(jnp.float32))
  return acc


class DenseFp32AccumulationTest(unittest.TestCase):

  def test_output_is_single_rounding_of_fp32_dot(self):
    rng = np.random.default_rng(0)
    x = np.asarray(jnp.asarray(rng.standard_normal((2, 7, 256)), jnp.bfloat16).astype(jnp.float32))
    k = np.asarray(jnp.asarray(rng.standard_normal((256, 32)), jnp.bfloat16).astype(jnp.float32))
    layer = _dense(True)
    layer.kernel.value = jnp.asarray(k)
    out = layer(jnp.asarray(x, jnp.bfloat16))
    self.assertEqual(out.dtype, jnp.bfloat16)
    once = np.asarray(jnp.asarray((x.astype(np.float64) @ k.astype(np.float64)).astype(np.float32), jnp.bfloat16))
    mismatch = np.mean(np.asarray(out, np.float32) != once.astype(np.float32))
    self.assertLess(mismatch, 0.01)  # fp32 summation order vs float64 can flip a rare rounding tie
    # the windowed-matmul style (bf16 partial sums) is measurably further from the single rounding
    windowed = _emulated_windowed(x, k, 4)
    self.assertGreater(np.mean(windowed != once.astype(np.float32)), 5 * mismatch + 0.01)

  def test_default_unchanged_on_single_device(self):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((3, 256)), jnp.bfloat16)
    k = jnp.asarray(jnp.asarray(rng.standard_normal((256, 32)), jnp.bfloat16).astype(jnp.float32))
    outs = []
    for flag in (False, True):
      layer = _dense(flag)
      layer.kernel.value = k
      outs.append(np.asarray(layer(x), np.float32))
    np.testing.assert_array_equal(outs[0], outs[1])  # one un-windowed dot already accumulates in fp32


if __name__ == "__main__":
  unittest.main()
