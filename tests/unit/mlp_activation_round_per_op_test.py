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

"""Gated MLP activation rounding: activations_in_float32 (one rounding) and mlp_activation_round_per_op
(one rounding per op), for the routed-expert path (RoutedMoE.apply_ffn_activation) and MlpBlock."""

import sys
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers import linears, moe
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

F32, BF16 = jnp.float32, jnp.bfloat16


def _gate_up(shape=(64, 96), seed=0):
  rng = np.random.default_rng(seed)
  return (jnp.asarray(rng.standard_normal(shape) * 4, BF16), jnp.asarray(rng.standard_normal(shape) * 4, BF16))


def _per_op(g, u):
  return ((jax.nn.silu(g.astype(F32)).astype(BF16).astype(F32)) * u.astype(F32)).astype(BF16)


def _once(g, u):
  return (jax.nn.silu(g.astype(F32)) * u.astype(F32)).astype(BF16)


class RoutedActivationTest(unittest.TestCase):

  def _apply(self, per_op, in_f32, g, u):
    cfg = types.SimpleNamespace(mlp_activation_round_per_op=per_op, activations_in_float32=in_f32,
                                decoder_block=DecoderBlockType.DEEPSEEK, mlp_activations_limit=0.0)
    fake = types.SimpleNamespace(config=cfg, activation_fn=jax.nn.silu, dtype=BF16)
    return moe.RoutedMoE.apply_ffn_activation(fake, g, u)

  def test_per_op(self):
    g, u = _gate_up()
    out = self._apply(True, False, g, u)
    self.assertEqual(out.dtype, BF16)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(_per_op(g, u), np.float32))
    # per-op wins over activations_in_float32
    np.testing.assert_array_equal(np.asarray(self._apply(True, True, g, u), np.float32), np.asarray(out, np.float32))

  def test_per_op_under_jit(self):
    g, u = _gate_up()
    out = jax.jit(lambda a, b: self._apply(True, False, a, b))(g, u)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(_per_op(g, u), np.float32))

  def test_activations_in_float32(self):
    g, u = _gate_up()
    out = self._apply(False, True, g, u)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(_once(g, u), np.float32))

  def test_three_modes_differ(self):
    g, u = _gate_up()
    a = np.asarray(self._apply(False, False, g, u), np.float32)
    b = np.asarray(self._apply(False, True, g, u), np.float32)
    c = np.asarray(self._apply(True, False, g, u), np.float32)
    self.assertFalse(np.array_equal(b, c))
    self.assertEqual(a.shape, c.shape)


class MlpBlockActivationTest(unittest.TestCase):

  def _block(self, **kw):
    cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], per_device_batch_size=1.0, run_name="test",
                              enable_checkpointing=False, max_target_length=16, fused_mlp=False, dtype="bfloat16",
                              mlp_activations=["silu", "linear"], **kw)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    return linears.MlpBlock(config=cfg, mesh=mesh, in_features=32, intermediate_dim=48, dtype=BF16,
                            activations=["silu", "linear"], rngs=nnx.Rngs(params=0, dropout=1))

  def _check(self, ref_fn, **kw):
    blk = self._block(**kw)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((2, 16, 32)), BF16)
    out = blk(x, deterministic=True)
    h = ref_fn(blk.wi_0(x), blk.wi_1(x))
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(blk.wo(h), np.float32))

  def test_per_op(self):
    self._check(_per_op, mlp_activation_round_per_op=True)

  def test_activations_in_float32(self):
    self._check(_once, activations_in_float32=True)


if __name__ == "__main__":
  unittest.main()
