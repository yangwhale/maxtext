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

"""rmsnorm_scale_grad_fp32 / logits_dense_wgrad_fp32 / embedding_grad_fp32: forward unchanged, gradient stays float32.

Without the switches, the gradients of these float32 parameters are first produced in the compute dtype (bf16) and only
then cast back to float32: the transpose of reduce_precision rounds the RMSNorm scale gradient, the bf16 kernel copy of
a DenseGeneral gets a bf16 weight gradient, and the bf16 embedding table copy gets a bf16 scatter-add.
"""

import sys
import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.layers import embeddings, linears
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils, maxtext_utils
from tests.utils.test_helpers import get_test_config_path

BF16 = ml_dtypes.bfloat16


def _bf16_representable(a):
  a = np.asarray(a, np.float32)
  return np.array_equal(a, a.astype(BF16).astype(np.float32))


class RoundToDtypeStraightThroughTest(unittest.TestCase):

  def test_forward_rounds_gradient_passes_through(self):
    x = jnp.asarray([1.0 + 2.0**-12, 3.0, -0.3], jnp.float32)
    w = jnp.asarray([1.00001, 2.0, 3.0], jnp.float32)
    y = max_utils.round_to_dtype_straight_through(x, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x.astype(jnp.bfloat16).astype(jnp.float32)))
    g = jax.grad(lambda v: jnp.sum(max_utils.round_to_dtype_straight_through(v, jnp.bfloat16) * w))(x)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(w))
    # plain reduce_precision rounds the cotangent too
    g0 = jax.grad(lambda v: jnp.sum(jax.lax.reduce_precision(v, 8, 7) * w))(x)
    self.assertFalse(np.array_equal(np.asarray(g0), np.asarray(w)))


class RmsNormScaleGradFp32Test(unittest.TestCase):
  D = 256

  def _grad(self, flag):
    """Forward output, scale gradient, and its float64 reference (bf16 cotangent, exact sum over tokens)."""
    rng = np.random.default_rng(0)
    x = jnp.asarray((rng.standard_normal((512, self.D)) * 3).astype(BF16))
    w = jnp.asarray(rng.standard_normal((512, self.D)).astype(np.float32))
    n = RMSNorm(
        self.D,
        epsilon=1e-5,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        single_rounding=True,
        scale_grad_fp32=flag,
        rngs=nnx.Rngs(params=0),
    )
    n.scale.value = jnp.asarray((1 + 0.1 * rng.standard_normal(self.D)).astype(np.float32))

    def loss(m):
      return jnp.sum(m(x).astype(jnp.float32) * w)

    y = np.asarray(n(x).astype(jnp.float32))
    g = nnx.grad(loss)(n).scale.value
    x64 = np.asarray(x.astype(jnp.float32), np.float64)
    normed = x64 / np.sqrt(np.mean(x64 * x64, -1, keepdims=True) + 1e-5)
    ref = np.sum(normed * np.asarray(w.astype(jnp.bfloat16).astype(jnp.float32), np.float64), 0)  # bf16 cotangent
    return y, np.asarray(g, np.float32), ref

  def test_forward_unchanged_and_scale_grad_fp32(self):
    y0, g0, ref = self._grad(False)
    y1, g1, _ = self._grad(True)
    np.testing.assert_array_equal(y0, y1)
    self.assertTrue(_bf16_representable(g0))  # default: the scale gradient was rounded to bf16
    self.assertFalse(_bf16_representable(g1))
    e0 = np.linalg.norm(g0 - ref) / np.linalg.norm(ref)
    e1 = np.linalg.norm(g1 - ref) / np.linalg.norm(ref)
    self.assertLess(e1, 1e-5)
    self.assertGreater(e0, 20 * e1)


class DenseKernelGradFp32Test(unittest.TestCase):

  def _run(self, flag):
    """Forward output, kernel gradient, and its float64 reference."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((2, 64, 128)).astype(BF16))
    w = jnp.asarray(rng.standard_normal((2, 64, 48)).astype(np.float32))
    layer = linears.DenseGeneral(
        in_features_shape=128,
        out_features_shape=48,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        kernel_grad_fp32=flag,
        rngs=nnx.Rngs(params=0),
    )
    layer.kernel.value = jnp.asarray(rng.standard_normal((128, 48)).astype(np.float32))

    def loss(m):
      return jnp.sum(m(x).astype(jnp.float32) * w)

    out = np.asarray(layer(x).astype(jnp.float32))
    g = np.asarray(nnx.grad(loss)(layer).kernel.value, np.float32)
    ct = np.asarray(w.astype(jnp.bfloat16).astype(jnp.float32), np.float64).reshape(-1, 48)
    ref = np.asarray(x.astype(jnp.float32), np.float64).reshape(-1, 128).T @ ct
    return out, g, ref

  def test_forward_unchanged_and_kernel_grad_fp32(self):
    o0, g0, ref = self._run(False)
    o1, g1, _ = self._run(True)
    self.assertLess(np.mean(o0 != o1), 0.01)  # float32 summation order can flip a rare rounding tie
    self.assertTrue(_bf16_representable(g0))  # default: wgrad produced in bf16
    self.assertFalse(_bf16_representable(g1))
    e0 = np.linalg.norm(g0 - ref) / np.linalg.norm(ref)
    e1 = np.linalg.norm(g1 - ref) / np.linalg.norm(ref)
    self.assertLess(e1, 1e-5)
    self.assertGreater(e0, 20 * e1)


class EmbeddingGradFp32Test(unittest.TestCase):

  def _cfg(self, flag):
    argv = [sys.argv[0], get_test_config_path()]
    return pyconfig.initialize(
        argv,
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        embedding_grad_fp32=flag,
    )

  def _run(self, flag):
    """Forward output, table gradient, and its float64 reference (exact scatter-add of bf16 cotangents)."""
    cfg = self._cfg(flag)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    layer = embeddings.Embed(
        num_embeddings=16, num_features=32, config=cfg, mesh=mesh, dtype=jnp.bfloat16, rngs=nnx.Rngs(params=0)
    )
    rng = np.random.default_rng(2)
    layer.embedding.value = jnp.asarray(rng.standard_normal((16, 32)).astype(np.float32))
    ids = jnp.asarray(rng.integers(0, 4, (4, 128)), jnp.int32)  # few ids: many tokens per row
    w = jnp.asarray(rng.standard_normal((4, 128, 32)).astype(np.float32))

    def loss(m):
      return jnp.sum(m(ids).astype(jnp.float32) * w)

    out = np.asarray(layer(ids).astype(jnp.float32))
    g = np.asarray(nnx.grad(loss)(layer).embedding.value, np.float32)
    ct = np.asarray(w.astype(jnp.bfloat16).astype(jnp.float32), np.float64).reshape(-1, 32)
    ref = np.zeros((16, 32))
    np.add.at(ref, np.asarray(ids).reshape(-1), ct)
    return out, g, ref

  def test_forward_unchanged_and_table_grad_fp32(self):
    o0, g0, ref = self._run(False)
    o1, g1, _ = self._run(True)
    np.testing.assert_array_equal(o0, o1)
    e0 = np.linalg.norm(g0 - ref) / np.linalg.norm(ref)
    e1 = np.linalg.norm(g1 - ref) / np.linalg.norm(ref)
    self.assertLess(e1, 1e-6)
    self.assertGreater(e0, 20 * e1)


if __name__ == "__main__":
  unittest.main()
