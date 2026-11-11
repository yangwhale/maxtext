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

"""Muon partitioning switches for Hy4, all default off:

- muon_attention_gate_whole_matrix: gate_proj [emb, (layers), heads, v_head_dim] as one [emb, heads * v_head_dim] matrix.
- muon_exclude_ihc: hyper-connection matrices go to AdamW.
- muon_fuse_glu: wi_0 and wi_1 orthogonalized concatenated on the output axis.
With the switches off, every parameter keeps the partition it had before.
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.contrib import MuonDimensionNumbers as mdn
from optax.contrib._muon import muon as optax_muon

from maxtext.optimizers.optimizers import fuse_glu_for_muon
from maxtext.utils import muon_utils


class TransformLogicTest(unittest.TestCase):

  def test_gate_proj_whole_matrix(self):
    path = ("decoder", "moe_layers", "self_attention", "gate_proj", "kernel")
    self.assertEqual(
        muon_utils.transform_logic(path, (256, 2, 4, 32), attention_gate_head_shape=(4, 32)), mdn((0,), (-2, -1))
    )
    unscanned = ("decoder", "moe_layers_remainder", "self_attention", "gate_proj", "kernel")
    self.assertEqual(
        muon_utils.transform_logic(unscanned, (256, 4, 32), attention_gate_head_shape=(4, 32)), mdn((0,), (-2, -1))
    )
    # Without the head shape (no MLA elementwise gate in the config) the old rule applies.
    self.assertEqual(muon_utils.transform_logic(path, (256, 2, 4, 32)), mdn((0,), (-1,)))
    # A 2D gate (e.g. compressed attention, scanned: [emb, layers, proj]) is not head-expanded.
    self.assertEqual(muon_utils.transform_logic(path, (256, 2, 64), attention_gate_head_shape=(4, 32)), mdn((0,), (-1,)))

  def test_exclude_ihc(self):
    for path, shape in [
        (("decoder", "moe_layers", "mhc_attention", "pre_alpha"), (1024, 2, 4)),
        (("decoder", "dense_layers", "mhc_mlp", "post_alpha"), (1024, 1, 4)),
        (("decoder", "hc_head", "hc_fn"), (1024, 4)),
    ]:
      self.assertEqual(muon_utils.transform_logic(path, shape), mdn((0,), (-1,)))  # default: Muon
      self.assertIsNone(muon_utils.transform_logic(path, shape, exclude_hyper_connections=True))
    # Other weights are not affected by the switch.
    p = ("decoder", "moe_layers", "self_attention", "wq_a", "kernel")
    self.assertEqual(muon_utils.transform_logic(p, (256, 2, 64), exclude_hyper_connections=True), mdn((0,), (-1,)))


class Hy4TinyDefaultUnchangedTest(unittest.TestCase):
  """hy4-tiny: with the switches off nothing changes; muon_attention_gate_whole_matrix changes only gate_proj."""

  def _diff(self, *overrides):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel
    from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=import-outside-toplevel

    cfg = pyconfig.initialize(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "model_name=hy4-tiny",
            "scan_layers=True",
            "attention=dot_product",
            "opt_type=muon",
            "skip_jax_distributed_system=True",
            "run_name=muon_hy4_switches_test",
            *overrides,
        ]
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    _, model = model_creation_utils.create_nnx_abstract_model(cfg, mesh)
    _, params, _ = nnx.split(model, nnx.Param, ...)
    new = nnx.to_pure_dict(muon_utils.get_muon_weight_dimension_numbers(model, cfg))
    changed, count = [], [0]

    def walk(sh, md, path):
      if isinstance(sh, dict):
        for k in sh:
          walk(sh[k], md[k], path + (k,))
        return
      count[0] += 1
      if muon_utils.transform_logic(path, tuple(sh.shape), include_routers=True) != md:
        changed.append(path)

    walk(nnx.to_pure_dict(params), new, ())
    self.assertGreater(count[0], 200)
    return cfg, changed

  def test_defaults_change_nothing(self):
    cfg, changed = self._diff()
    self.assertFalse(cfg.muon_attention_gate_whole_matrix or cfg.muon_exclude_ihc or cfg.muon_fuse_glu)
    self.assertEqual(changed, [])

  def test_gate_switch_changes_only_gate(self):
    _, changed = self._diff("muon_attention_gate_whole_matrix=true")
    self.assertTrue(changed)
    for path in changed:
      self.assertIn("gate_proj", path)


def _toy():
  k = jax.random.split(jax.random.PRNGKey(0), 6)
  params = {
      "mlp": {
          "wi_0": {"kernel": jax.random.normal(k[0], (8, 6))},
          "wi_1": {"kernel": jax.random.normal(k[1], (8, 6))},
          "wo": {"kernel": jax.random.normal(k[2], (6, 8))},
      },
      "moe": {"wi_0": jax.random.normal(k[3], (3, 8, 5)), "wi_1": jax.random.normal(k[4], (3, 8, 5))},
      "norm": {"scale": jnp.ones((8,))},
  }
  grads = jax.tree_util.tree_map(lambda x: jax.random.normal(k[5], x.shape), params)
  m = {
      "mlp": {
          "wi_0": {"kernel": mdn((0,), (-1,))},
          "wi_1": {"kernel": mdn((0,), (-1,))},
          "wo": {"kernel": mdn((0,), (-1,))},
      },
      "moe": {"wi_0": mdn((-2,), (-1,)), "wi_1": mdn((-2,), (-1,))},
      "norm": {"scale": None},
  }
  return params, grads, m


class FuseGluTest(unittest.TestCase):

  def _kw(self):
    return {"learning_rate": 0.1, "weight_decay": 0.01, "consistent_rms": 0.2}

  def test_matches_concatenated_reference(self):
    params, grads, m = _toy()
    tx = fuse_glu_for_muon(lambda t: optax_muon(muon_weight_dimension_numbers=t, **self._kw()), m)
    state = tx.init(params)
    up, _ = tx.update(grads, state, params)
    # Reference: Muon on a tree where each pair is one concatenated parameter.
    cat = lambda a, b: jnp.concatenate([a, b], axis=-1)
    p2 = {
        "mlp_fc1": cat(params["mlp"]["wi_0"]["kernel"], params["mlp"]["wi_1"]["kernel"]),
        "wo": params["mlp"]["wo"]["kernel"],
        "moe_fc1": cat(params["moe"]["wi_0"], params["moe"]["wi_1"]),
        "scale": params["norm"]["scale"],
    }
    g2 = {
        "mlp_fc1": cat(grads["mlp"]["wi_0"]["kernel"], grads["mlp"]["wi_1"]["kernel"]),
        "wo": grads["mlp"]["wo"]["kernel"],
        "moe_fc1": cat(grads["moe"]["wi_0"], grads["moe"]["wi_1"]),
        "scale": grads["norm"]["scale"],
    }
    m2 = {"mlp_fc1": mdn((0,), (-1,)), "wo": mdn((0,), (-1,)), "moe_fc1": mdn((-2,), (-1,)), "scale": None}
    tx2 = optax_muon(muon_weight_dimension_numbers=m2, **self._kw())
    up2, _ = tx2.update(g2, tx2.init(p2), p2)
    np.testing.assert_allclose(
        cat(up["mlp"]["wi_0"]["kernel"], up["mlp"]["wi_1"]["kernel"]), up2["mlp_fc1"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(cat(up["moe"]["wi_0"], up["moe"]["wi_1"]), up2["moe_fc1"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(up["mlp"]["wo"]["kernel"], up2["wo"], rtol=1e-6)
    np.testing.assert_allclose(up["norm"]["scale"], up2["scale"], rtol=1e-6)
    # And it really differs from orthogonalizing the two halves separately.
    sep = optax_muon(muon_weight_dimension_numbers=m, **self._kw())
    up3, _ = sep.update(grads, sep.init(params), params)
    self.assertFalse(np.allclose(up["mlp"]["wi_0"]["kernel"], up3["mlp"]["wi_0"]["kernel"], atol=1e-4))

  def test_multi_step_shapes(self):
    params, grads, m = _toy()
    tx = fuse_glu_for_muon(lambda t: optax_muon(muon_weight_dimension_numbers=t, **self._kw()), m)
    state = tx.init(params)
    for _ in range(3):
      up, state = tx.update(grads, state, params)
      params = optax.apply_updates(params, up)
    jax.tree_util.tree_map(lambda a, b: self.assertEqual(a.shape, b.shape), params, _toy()[0])


if __name__ == "__main__":
  unittest.main()
