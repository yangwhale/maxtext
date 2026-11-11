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

"""debug_routing_digest / routing_debug.count_reselections."""

import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils, routing_debug
from tests.utils.test_helpers import get_test_config_path


class CountTest(unittest.TestCase):

  def test_counts(self):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 16, (64, 4))
    b = rng.integers(0, 16, (64, 4))  # another layer: agrees on almost no token
    a2 = a.copy()
    a2[3] = [15, 14, 13, 12] if set(a[3]) != {12, 13, 14, 15} else [0, 1, 2, 3]  # one token re-selected
    r = routing_debug.count_reselections([a, b, a2, b.copy()])
    self.assertEqual(r["calls"], 4)
    self.assertEqual(r["repeats"], 2)
    self.assertEqual(r["slots_compared"], 2 * a.size)
    self.assertGreaterEqual(r["slots_differ"], 1)
    self.assertLessEqual(r["slots_differ"], 4)
    self.assertEqual(routing_debug.count_reselections([a, b])["repeats"], 0)


class RematTest(unittest.TestCase):

  def test_full_remat_recomputes_the_router_and_logs_it(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="routing_debug_test",
        enable_checkpointing=False,
        dtype="float32",
        max_target_length=8,
        per_device_batch_size=1,
        num_experts=4,
        num_experts_per_tok=2,
        base_emb_dim=16,
        base_moe_mlp_dim=32,
        base_mlp_dim=32,
        sparse_matmul=True,
        megablox=False,
        ici_fsdp_parallelism=-1,
        debug_routing_digest=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=4,
        num_experts_per_tok=2,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        intermediate_dim=32,
        dtype=jnp.float32,
        rngs=nnx.Rngs(params=0),
    )
    gd, params, rest = nnx.split(model, nnx.Param, ...)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((jax.device_count(), 8, 16)), jnp.float32)

    @jax.checkpoint
    def f(p, x_):
      return jnp.sum(nnx.merge(gd, p, rest)(x_)[0] ** 2)

    routing_debug.LOG.clear()
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      g = jax.grad(f)(params, x)
    jax.block_until_ready(g)
    jax.effects_barrier()
    r = routing_debug.count_reselections()
    self.assertGreater(r["calls"], 0)
    self.assertEqual(r["repeats"] * 2, r["calls"])  # forward + recompute in the backward
    self.assertEqual(r["slots_differ"], 0)


if __name__ == "__main__":
  unittest.main()
