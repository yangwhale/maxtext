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

"""hy4-mid differs from hy4-mini only in width, heads, q_lora, dense FFN and expert width; the MaxText yml and the HF
config agree, and the model name is registered for conversion."""

import os
import unittest

import yaml

from maxtext.checkpoint_conversion.utils import hf_model_configs, param_mapping
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

_DIFF = {
    "base_emb_dim": 6144,
    "base_mlp_dim": 18432,
    "base_moe_mlp_dim": 512,
    "base_num_query_heads": 64,
    "base_num_kv_heads": 64,
    "q_lora_rank": 2048,
}


def _load(name):
  with open(os.path.join(MAXTEXT_CONFIGS_DIR, "models", f"{name}.yml"), encoding="utf-8") as f:
    return yaml.safe_load(f)


class Hy4MidConfigTest(unittest.TestCase):

  def test_yml_differs_from_mini_only_in_dims(self):
    mini, mid = _load("hy4-mini"), _load("hy4-mid")
    self.assertEqual(set(mini), set(mid))
    for k, v in mid.items():
      self.assertEqual(v, _DIFF.get(k, mini[k]), k)

  def test_hf_config_matches_yml(self):
    hf = hf_model_configs.hy4_mid_dict
    mid = _load("hy4-mid")
    self.assertEqual(hf["hidden_size"], mid["base_emb_dim"])
    self.assertEqual(hf["intermediate_size"], mid["base_mlp_dim"])
    self.assertEqual(hf["moe_intermediate_size"], mid["base_moe_mlp_dim"])
    self.assertEqual(hf["num_attention_heads"], mid["base_num_query_heads"])
    self.assertEqual(hf["q_lora_rank"], mid["q_lora_rank"])
    self.assertEqual(hf["num_hidden_layers"], mid["base_num_decoder_layers"])
    self.assertEqual(hf["n_routed_experts"], mid["num_experts"])
    self.assertEqual(hf["num_experts_per_tok"], mid["num_experts_per_tok"])

  def test_registered(self):
    self.assertIn("hy4-mid", hf_model_configs.HF_MODEL_CONFIGS)
    self.assertIn("hy4-mid", param_mapping.PARAM_MAPPING)
    self.assertIn("hy4-mid", param_mapping.HOOK_FNS)


if __name__ == "__main__":
  unittest.main()
