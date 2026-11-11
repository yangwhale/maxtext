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

"""embedding_shard_over_data: only the token-embedding table is also sharded over 'data'.

Run on CPU with XLA_FLAGS=--xla_force_host_platform_device_count=8 to get a data-parallel mesh.
"""

import os
import re
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.core.spmd import logical_axis_rules as flax_rules
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.configs import pyconfig
from maxtext.layers.embeddings import Embed, _spec_axes
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.utils.sharding import logical_to_mesh_axes
from tests.utils.test_helpers import get_test_config_path

VOCAB, EMB, SEQ = 64, 16, 32


def _rule(rules, name):
  return next((tuple(r[1]) for r in rules if r and r[0] == name), None)


class TestEmbeddingShardOverDataConfig(unittest.TestCase):

  def _config(self, **kw):
    return pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        **kw,
    )

  def test_off_adds_no_rule(self):
    self.assertIsNone(_rule(self._config().logical_axis_rules, "embed_vocab_table"))

  def test_on_maps_data_then_embed_vocab_axes(self):
    cfg = self._config(embedding_shard_over_data=True)
    embed_vocab = _rule(cfg.logical_axis_rules, "embed_vocab")
    self.assertEqual(_rule(cfg.logical_axis_rules, "embed_vocab_table"), ("data", *embed_vocab))
    # the output head and other tables keep using embed_vocab, which is unchanged
    self.assertNotIn("data", embed_vocab)

  def test_rejects_zero1(self):
    with self.assertRaisesRegex(ValueError, "embedding_shard_over_data"):
      self._config(embedding_shard_over_data=True, shard_optimizer_over_data=True)


class TestEmbeddingShardOverDataPartitioning(unittest.TestCase):
  """Pure data parallelism over every local device; the table is sharded over 'data' only when the switch is on."""

  def setUp(self):
    self.n = jax.device_count()
    if self.n < 2:
      self.skipTest("needs a data-parallel mesh of at least 2 devices")

  def _cfg_mesh(self, on):
    cfg = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
        run_name="test",
        enable_checkpointing=False,
        ici_data_parallelism=self.n,
        ici_fsdp_parallelism=1,
        embedding_shard_over_data=on,
    )
    return cfg, Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)

  def _embed(self, on):
    cfg, mesh = self._cfg_mesh(on)
    with jax.set_mesh(mesh), flax_rules(cfg.logical_axis_rules):
      emb = Embed(VOCAB, EMB, cfg, mesh, dtype=jnp.bfloat16, rngs=nnx.Rngs(params=0))
    return cfg, mesh, emb

  def test_param_logical_and_mesh_axes(self):
    for on, name in ((False, "embed_vocab"), (True, "embed_vocab_table")):
      cfg, mesh, emb = self._embed(on)
      self.assertEqual(tuple(emb.embedding.get_metadata()["out_sharding"]), ("vocab", name))
      spec = logical_to_mesh_axes(("vocab", name), mesh, rules=cfg.logical_axis_rules)
      self.assertEqual("data" in _spec_axes(spec[1]), on)

  def _grad_hlo(self, on):
    """Gradient of sum(embed(ids) * g) with the table placed as its logical axes say; returns (grad, optimized HLO)."""
    cfg, mesh, emb = self._embed(on)
    graphdef, state = nnx.split(emb)
    spec = logical_to_mesh_axes(tuple(emb.embedding.get_metadata()["out_sharding"]), mesh, rules=cfg.logical_axis_rules)
    rng = np.random.default_rng(0)
    table = jax.device_put(jnp.asarray(rng.normal(size=(VOCAB, EMB)), jnp.float32), NamedSharding(mesh, spec))
    ids = jax.device_put(jnp.asarray(rng.integers(0, 6, size=(self.n, SEQ)), jnp.int32), NamedSharding(mesh, P("data")))
    g = jax.device_put(
        jnp.asarray(rng.normal(size=(self.n, SEQ, EMB)), jnp.bfloat16), NamedSharding(mesh, P("data", None, None))
    )

    def loss(t, ids, g):
      m = nnx.merge(graphdef, jax.tree.map(lambda _: t, state))
      return jnp.sum(m(ids).astype(jnp.float32) * g.astype(jnp.float32))

    f = jax.jit(jax.grad(loss), out_shardings=NamedSharding(mesh, spec))
    with jax.set_mesh(mesh), flax_rules(cfg.logical_axis_rules):
      grad = f(table, ids, g)
      hlo = f.lower(table, ids, g).compile().as_text()
    return grad, hlo, np.asarray(ids), np.asarray(g, np.float32)

  def test_gradient_value_is_the_scatter_add(self):
    for on in (False, True):
      grad, _, ids, g = self._grad_hlo(on)
      exact = np.zeros((VOCAB, EMB), np.float64)
      np.add.at(exact, ids.reshape(-1), g.reshape(-1, EMB))
      # bfloat16 accumulation of a handful of updates per row
      np.testing.assert_allclose(np.asarray(grad, np.float64), exact, rtol=0, atol=0.25)

  def test_no_full_table_all_reduce_when_on(self):
    _, hlo_off, _, _ = self._grad_hlo(False)
    _, hlo_on, _, _ = self._grad_hlo(True)
    self.assertTrue(re.search(rf"\[{VOCAB},{EMB}\]\S* all-reduce\(", hlo_off), "default path sums the full table")
    self.assertIsNone(re.search(rf"\[{VOCAB},{EMB}\]\S* all-reduce\(", hlo_on), "sharded table must not all-reduce it")


if __name__ == "__main__":
  unittest.main()
