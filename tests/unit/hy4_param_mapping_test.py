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

"""Layout checks for the Hy4 HF -> MaxText parameter mapping (IndexCache blocks, dense layers, MTP)."""

from types import SimpleNamespace
import unittest

import pytest

from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.hf_model_configs import hy4_preview_dict, hy4_tiny_dict

pytestmark = [pytest.mark.decoupled_target]

# Layers whose indexer_types entry is "full" in the tencent/Hy4-preview config.json.
HY4_PREVIEW_FULL_INDEXER_LAYERS = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77]


def _maxtext_config(**overrides):
  values = dict(use_indexer=True, indexer_share_interval=4, mtp_num_layers=1, first_num_dense_layers=1)
  return SimpleNamespace(**(values | overrides))


class Hy4ParamMappingTest(unittest.TestCase):

  def test_preview_indexer_types_match_the_hf_config(self):
    full = [i for i, t in enumerate(hy4_preview_dict["indexer_types"]) if t == "full"]
    self.assertEqual(full, HY4_PREVIEW_FULL_INDEXER_LAYERS)

  def test_index_cache_blocks(self):
    mapping = param_mapping.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(hy4_preview_dict, _maxtext_config(), scan_layers=True)
    wk = "self_attention-indexer-wk-kernel"
    # 77 MoE layers = 19 groups of 4 plus one remainder layer (77); only the first layer of each group has an indexer.
    self.assertEqual(
        mapping[f"params-decoder-moe_layers-layers_0-{wk}"],
        [f"model.layers.{i}.self_attn.indexer.wk.weight" for i in range(1, 77, 4)],
    )
    for j in (1, 2, 3):
      self.assertNotIn(f"params-decoder-moe_layers-layers_{j}-{wk}", mapping)
    self.assertEqual(
        mapping["params-decoder-moe_layers-layers_2-self_attention-out-kernel"],
        [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(3, 77, 4)],
    )
    self.assertEqual(
        mapping[f"params-decoder-moe_layers_remainder-layers_0-{wk}"], "model.layers.77.self_attn.indexer.wk.weight"
    )
    self.assertEqual(
        mapping[f"params-decoder-dense_layers-{wk}"], ["model.layers.0.self_attn.indexer.wk.weight"]
    )
    self.assertEqual(
        mapping[f"params-mtp_block-mtp_layer_1-mtp_1_transformer_layer-{wk}"],
        "model.mtp_layers.0.self_attn.indexer.wk.weight",
    )

  def test_rejects_indexer_layout_that_does_not_follow_groups(self):
    # One non-first layer of one group (layer 2) also runs an indexer; the same position in the other 18 groups
    # does not, so a check that only rejects the position when every group is "full" would let it through.
    config = dict(hy4_preview_dict, indexer_types=list(hy4_preview_dict["indexer_types"]))
    config["indexer_types"][2] = "full"
    with self.assertRaisesRegex(ValueError, "groups of 4"):
      param_mapping.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(config, _maxtext_config(), scan_layers=True)

  def test_rejects_dense_layer_mismatch(self):
    with self.assertRaisesRegex(ValueError, "first_num_dense_layers"):
      param_mapping.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(
          hy4_tiny_dict, _maxtext_config(first_num_dense_layers=2), scan_layers=True
      )
    config = dict(hy4_tiny_dict, mlp_layer_types=["sparse", "dense"] + ["sparse"] * 5)
    with self.assertRaisesRegex(ValueError, "leading"):
      param_mapping.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(config, _maxtext_config(), scan_layers=True)

  def test_rejects_fewer_moe_layers_than_one_group(self):
    config = dict(hy4_tiny_dict, num_hidden_layers=4, mlp_layer_types=["dense"] + ["sparse"] * 3)
    config["indexer_types"] = ["full", "full", "shared", "shared"]
    with self.assertRaises(NotImplementedError):
      param_mapping.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(config, _maxtext_config(), scan_layers=True)

  def test_hc_head_hook_uses_the_kernel_transform(self):
    hooks = param_mapping.HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN(hy4_tiny_dict, _maxtext_config(), scan_layers=True)
    self.assertIs(hooks["params-decoder-hc_head-hc_fn"], hooks["params-decoder-logits_dense-kernel"])


if __name__ == "__main__":
  unittest.main()
