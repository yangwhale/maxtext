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

"""Data parallelism + gradient accumulation + a float32 scatter-accumulated embedding gradient (embedding_grad_fp32).

On TPU (v5p, data parallelism 8, GA 2) XLA moved the data-parallel all-reduce of this gradient out of the accumulation
loop and zeroed the accumulator on partitions != 0 in every iteration, so only the last microbatch survived there
(0.51 of the embedding-gradient norm). gradient_accumulation now puts an optimization barrier on each microbatch
gradient when embedding_grad_fp32, data parallelism > 1 and GA > 1 all hold. This test checks the accumulated gradient
against the one-shot gradient on a data-parallel mesh of all local devices (on CPU run with
XLA_FLAGS=--xla_force_host_platform_device_count=8); the CPU compiler may not apply the same code motion, so the
regression itself shows on TPU.
"""

import unittest
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import ShardMode
from maxtext.utils import gradient_accumulation

VOCAB, EMB, SEQ = 64, 16, 8


@dataclass
class _Cfg:
  gradient_accumulation_steps: int = 2
  use_tunix_gradient_accumulation: bool = False
  shard_optimizer_over_data: bool = False
  shard_mode: int = ShardMode.AUTO
  ici_data_parallelism: int = 1
  debug_sharding: bool = False
  training_objective: str = "causal_lm"
  embedding_grad_fp32: bool = True


class _TinyEmbed(nnx.Module):
  """A float32 table gathered by token id and cast to bfloat16, like Embed with embedding_grad_fp32."""

  def __init__(self):
    self.table = nnx.Param(jnp.asarray(np.random.default_rng(0).normal(size=(VOCAB, EMB)), jnp.float32))
    self.w = jnp.asarray(np.random.default_rng(1).normal(size=(EMB,)), jnp.float32)

  def __call__(self, ids):
    # float32 end to end so the one-shot reference is exact up to summation order; the scatter-add in the backward is
    # the float32 one that embedding_grad_fp32 produces
    return self.table.get_value().at[ids].get()


def _loss_fn(model, config, data, dropout_rng, params, is_train=True):
  del config, dropout_rng, params, is_train
  per_token = (model(data["ids"]) * model.w).sum(-1) ** 2
  aux = {"xent_sum": per_token.sum(), "total_weights": jnp.asarray(per_token.size, jnp.float32),
         "moe_lb_loss": jnp.array(0.0), "indexer_loss": jnp.array(0.0), "mtp_loss": jnp.array(0.0)}
  return per_token.sum(), aux


class TestEmbeddingGradDataParallelGA(unittest.TestCase):

  def test_accumulated_embedding_grad_matches_one_shot(self):
    self._check(True)

  def test_without_the_switch_matches_too(self):
    self._check(False)

  def _check(self, grad_fp32):
    n = jax.device_count()
    if n < 2:
      self.skipTest("needs a data-parallel mesh of at least 2 devices")
    mesh = Mesh(np.array(jax.devices()), ("data",))
    model = _TinyEmbed()
    _, params, _ = nnx.split(model, nnx.Param, ...)
    shardings = jax.tree.map(lambda _: NamedSharding(mesh, PartitionSpec()), params)
    # several devices hit the same rows, and both microbatches touch rows the other one does not
    ids = jnp.asarray(np.random.default_rng(2).integers(0, VOCAB, size=(2 * n, SEQ)), jnp.int32)
    data = {"ids": jax.device_put(ids, NamedSharding(mesh, PartitionSpec("data")))}

    def step(model, data):
      _, _, grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
          _loss_fn, _Cfg(embedding_grad_fp32=grad_fp32), model, params=None, params_shardings=shardings, data=data, dropout_rng=None)
      return grads

    with jax.set_mesh(mesh):
      grads = nnx.jit(step)(model, data)

    table = model.table.get_value()

    def one_shot(t):
      per_token = (t.at[ids].get() * model.w).sum(-1) ** 2
      return per_token.sum() / per_token.size

    expected = jax.grad(one_shot)(table)
    got = jax.tree.leaves(grads)[0]
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
