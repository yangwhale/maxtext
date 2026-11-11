# Copyright 2026 Google LLC
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

"""embedding_grad_fp32_local_bf16_comm: float32 local scatter-add, one bfloat16 rounding, bfloat16 data-parallel psum.

Run on CPU with XLA_FLAGS=--xla_force_host_platform_device_count=8 to get a data-parallel mesh.
"""

import re
import unittest
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.common.common_types import ShardMode
from maxtext.layers.embeddings import _embed_local_fp32_bf16_comm
from maxtext.utils import gradient_accumulation

VOCAB, EMB, SEQ = 32, 16, 64


class TestEmbeddingLocalFp32Bf16Comm(unittest.TestCase):

  def setUp(self):
    self.n = jax.device_count()
    if self.n < 2:
      self.skipTest("needs a data-parallel mesh of at least 2 devices")
    self.mesh = Mesh(np.array(jax.devices()).reshape(self.n, 1), ("data", "tensor"))
    rng = np.random.default_rng(0)
    self.table = jnp.asarray(rng.normal(size=(VOCAB, EMB)), jnp.float32)
    # few distinct ids, many repeats per device: the case where per-update bfloat16 rounding loses precision
    self.ids = jnp.asarray(rng.integers(0, 4, size=(self.n, SEQ)), jnp.int32)
    self.g = jnp.asarray(rng.normal(size=(self.n, SEQ, EMB)), jnp.float32).astype(jnp.bfloat16)
    self.spec = P("data", None, "tensor")

  def _run(self, fn):
    ids = jax.device_put(self.ids, NamedSharding(self.mesh, P("data")))
    g = jax.device_put(self.g, NamedSharding(self.mesh, self.spec))
    with jax.set_mesh(self.mesh):
      return jax.jit(fn)(self.table, ids, g)

  def _new(self, table, ids, g):
    out, vjp = jax.vjp(lambda t: _embed_local_fp32_bf16_comm(t, ids, jnp.bfloat16, self.mesh, self.spec, None), table)
    return out, vjp(g)[0]

  def test_forward_is_bitwise_the_default(self):
    out, _ = self._run(self._new)
    expected = jnp.asarray(self.table, jnp.bfloat16).at[self.ids].get()
    self.assertEqual(out.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

  def test_gradient_is_float32_local_sum_rounded_once_then_bfloat16_sum(self):
    _, grad = self._run(self._new)
    self.assertEqual(grad.dtype, jnp.float32)
    g32 = np.asarray(self.g, np.float32)
    ids = np.asarray(self.ids)
    local = []
    for d in range(self.n):
      acc = np.zeros((VOCAB, EMB), np.float32)
      np.add.at(acc, ids[d], g32[d])
      local.append(np.asarray(jnp.asarray(acc, jnp.bfloat16), np.float32))  # one rounding per device
    exact = np.sum(local, axis=0)
    # the cross-device sum is in bfloat16: within n bfloat16 roundings of the exact sum of the rounded locals
    np.testing.assert_allclose(np.asarray(grad), exact, rtol=self.n * 2.0**-8, atol=1e-6)
    # and far closer to the float32 truth than a per-update bfloat16 scatter-add
    truth = np.zeros((VOCAB, EMB), np.float64)
    np.add.at(truth, ids.reshape(-1), g32.reshape(-1, EMB))
    per_update = jnp.zeros((VOCAB, EMB), jnp.bfloat16).at[self.ids.reshape(-1)].add(self.g.reshape(-1, EMB))
    err_new = np.linalg.norm(np.asarray(grad) - truth)
    err_bf16 = np.linalg.norm(np.asarray(per_update, np.float64) - truth)
    self.assertLess(err_new, err_bf16)

  def test_cross_device_all_reduce_is_bfloat16(self):
    ids = jax.device_put(self.ids, NamedSharding(self.mesh, P("data")))
    g = jax.device_put(self.g, NamedSharding(self.mesh, self.spec))
    with jax.set_mesh(self.mesh):
      lowered = jax.jit(lambda t, i, c: self._new(t, i, c)[1]).lower(self.table, ids, g)
    # the program handed to XLA: the psum operand is the bfloat16 local sum behind an optimization barrier
    text = lowered.as_text()
    reduces = re.findall(r"stablehlo.all_reduce\"\((%\w+)\)", text)
    self.assertEqual(len(reduces), 1, reduces)
    barrier = re.search(rf"{re.escape(reduces[0])} = stablehlo.optimization_barrier \S+ : tensor<{VOCAB}x{EMB}xbf16>", text)
    self.assertIsNotNone(barrier, "the all-reduce operand is not the bfloat16 barrier output")
    if jax.default_backend() == "tpu":
      # TPU keeps bfloat16 collectives (the CPU backend widens them to float32 itself, so check only there)
      compiled = lowered.compile().as_text()
      for shape in re.findall(r"= (\S+) all-reduce(?:-start)?\(", compiled):
        self.assertTrue(shape.startswith(f"bf16[{VOCAB},{EMB}]"), shape)


@dataclass
class _Cfg:
  gradient_accumulation_steps: int = 2
  use_tunix_gradient_accumulation: bool = False
  shard_optimizer_over_data: bool = False
  shard_mode: int = ShardMode.AUTO
  ici_data_parallelism: int = 1
  debug_sharding: bool = False
  training_objective: str = "causal_lm"
  embedding_grad_fp32: bool = False
  embedding_grad_fp32_local_bf16_comm: bool = True


class _TinyEmbed(nnx.Module):

  def __init__(self, mesh):
    self.mesh = mesh
    self.table = nnx.Param(jnp.asarray(np.random.default_rng(0).normal(size=(VOCAB, EMB)), jnp.float32))
    self.w = jnp.asarray(np.random.default_rng(1).normal(size=(EMB,)), jnp.float32)

  def __call__(self, ids):
    x = _embed_local_fp32_bf16_comm(self.table.get_value(), ids, jnp.bfloat16, self.mesh, P("data", None, None), None)
    return x.astype(jnp.float32)


def _loss_fn(model, config, data, dropout_rng, params, is_train=True):
  del config, dropout_rng, params, is_train
  per_token = (model(data["ids"]) * model.w).sum(-1) ** 2
  aux = {"xent_sum": per_token.sum(), "total_weights": jnp.asarray(per_token.size, jnp.float32),
         "moe_lb_loss": jnp.array(0.0), "indexer_loss": jnp.array(0.0), "mtp_loss": jnp.array(0.0)}
  return per_token.sum(), aux


class TestLocalFp32Bf16CommWithGradientAccumulation(unittest.TestCase):
  """GA 2 on a data-parallel mesh: the accumulated gradient is the sum of the two microbatch gradients (no microbatch
  lost, the d5c9b113 failure mode), each microbatch with its own bfloat16 all-reduce inside the loop."""

  def test_accumulated_equals_sum_of_microbatches(self):
    n = jax.device_count()
    if n < 2:
      self.skipTest("needs a data-parallel mesh of at least 2 devices")
    mesh = Mesh(np.array(jax.devices()), ("data",))
    model = _TinyEmbed(mesh)
    _, params, _ = nnx.split(model, nnx.Param, ...)
    shardings = jax.tree.map(lambda _: NamedSharding(mesh, P()), params)
    ids = jnp.asarray(np.random.default_rng(2).integers(0, VOCAB, size=(2 * n, SEQ)), jnp.int32)
    data = {"ids": jax.device_put(ids, NamedSharding(mesh, P("data")))}

    def step(model, data):
      _, _, grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
          _loss_fn, _Cfg(), model, params=None, params_shardings=shardings, data=data, dropout_rng=None)
      return grads

    def micro(table, ids_mb):
      def f(t):
        x = _embed_local_fp32_bf16_comm(t, ids_mb, jnp.bfloat16, mesh, P("data", None, None), None).astype(jnp.float32)
        return ((x * model.w).sum(-1) ** 2).sum()
      return jax.grad(f)(table)

    with jax.set_mesh(mesh):
      got = jax.tree.leaves(nnx.jit(step)(model, data))[0]
      table = model.table.get_value()
      # gradient_accumulation splits the global batch into microbatches of n rows each (row i of every device shard)
      mbs = ids.reshape(n, 2, SEQ).transpose(1, 0, 2)
      parts = [jax.jit(micro)(table, jax.device_put(mbs[k], NamedSharding(mesh, P("data")))) for k in range(2)]
    expected = (parts[0] + parts[1]) / (2 * n * SEQ)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
  unittest.main()
