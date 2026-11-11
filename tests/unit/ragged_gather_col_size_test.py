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

"""Tests for the ragged gather kernel's column tile size (calculate_col_size)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu

from maxtext.kernels.ragged import ragged_gather


def _info(chip_version):
  # The TpuInfo jax reports for that chip (one TensorCore per device): chip_version is a ChipVersion enum with string
  # values, generation an int.
  return pltpu.get_tpu_info_for_chip(chip_version, 1)


class CalculateColSizeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # v7x: 16 SparseCore lanes x 6144 x 4 bytes = 384 KiB fits the 512 KiB * 0.9 v7 budget -> one full-width tile.
      ("v7x_hidden_6144", pltpu.ChipVersion.TPU_7X, 6144, 6144),
      # v6e: 8 lanes x 6144 x 4 = 192 KiB fits 256 KiB * 0.9 -> one tile.
      ("v6e_hidden_6144", pltpu.ChipVersion.TPU_V6E, 6144, 6144),
      # v5p takes the 128 KiB default either way (unchanged by the generation match): 8 x 6144 x 4 = 192 KiB needs two
      # tiles of 3072. Kept as the case the old chip_version match also passed.
      ("v5p_hidden_6144", pltpu.ChipVersion.TPU_V5P, 6144, 3072),
  )
  def test_col_size_per_generation(self, chip_version, hidden_size, expected):
    with mock.patch.object(ragged_gather.pltpu, "get_tpu_info", return_value=_info(chip_version)):
      self.assertEqual(ragged_gather.calculate_col_size(hidden_size), expected)


if __name__ == "__main__":
  absltest.main()
