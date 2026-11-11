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

"""Hy4 model configs load, keep their float32 precision settings, and follow the HF init."""

import math
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path


def _config(model_name, **kw):
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="hy4_config_test",
      enable_checkpointing=False,
      model_name=model_name,
      per_device_batch_size=1,
      max_target_length=32,
      max_prefill_predict_length=32,
      **kw,
  )


class Hy4ConfigTest(unittest.TestCase):

  def test_hy4_preview_loads_with_precision_settings(self):
    cfg = _config("hy4-preview", attention="dot_product")
    self.assertEqual(cfg.num_decoder_layers, 78)
    self.assertEqual((cfg.num_experts, cfg.num_experts_per_tok, cfg.mtp_num_layers), (256, 8, 1))
    self.assertTrue(cfg.use_indexer and cfg.indexer_sparse_training)
    self.assertEqual(cfg.indexer_share_interval, 4)
    self.assertTrue(cfg.float32_gate_logits)
    self.assertEqual(cfg.router_matmul_precision, "highest")
    self.assertEqual(cfg.hc_matmul_precision, "highest")
    self.assertEqual(cfg.moe_routing_maps, "device")
    self.assertTrue(cfg.logits_dot_in_fp32)

  def test_hy4_tiny_follows_hf_init(self):
    cfg = _config("hy4-tiny", weight_dtype="float32", dtype="float32", attention="dot_product")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    flat = {"/".join(map(str, p)): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    leaf = lambda k: k.rsplit("/", 1)[-1]
    matrices = {k: v for k, v in flat.items() if leaf(k) in ("kernel", "embedding", "wi_0", "wi_1", "wo", "pre_alpha",
                                                             "post_alpha", "hc_fn") and v.size >= 4096}
    for name in ("embedding", "wi_0", "wi_1", "wo", "pre_alpha", "post_alpha", "hc_fn"):
      self.assertTrue(any(leaf(k) == name for k in matrices), name)
    for k, v in matrices.items():
      self.assertAlmostEqual(float(v.std()), cfg.init_weights_std, delta=0.05 * cfg.init_weights_std, msg=k)
    # Norm scales keep their own init (ones), not the re-draw.
    scales = [v for k, v in flat.items() if leaf(k) == "scale" and "norm" in k]
    self.assertTrue(scales and all(np.all(v == 1.0) for v in scales))
    np.testing.assert_allclose(flat["decoder/hc_head/hc_base"], -math.log(cfg.mhc_expansion_rate - 1), rtol=1e-6)


if __name__ == "__main__":
  unittest.main()
