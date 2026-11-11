# Branch `hunyuan`: Hunyuan 3 on MaxText

This branch is upstream [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext) plus the changes we
used to train Hunyuan 3 (Hy3, 295B-A21B) on TPU and to align its numerics with the GPU (Megatron) reference.
It is published as a single commit on top of the upstream commit it was developed against, so
`git diff HEAD~1` shows the whole change set.

The numerical alignment work that drove these changes (method, golden runs, per-level checks, results) is in
[yangwhale/janus](https://github.com/yangwhale/janus), under `models/hunyuan3/`. In the janus documents, "the
`hunyuan` branch of maxtext" means this branch.

## Model

- `decoder_block: hunyuan3` (`src/maxtext/models/hunyuan3.py`): dense first layer, MoE layers with sigmoid routing and
  expert bias, shared expert, MTP.
- Configs: `src/maxtext/configs/models/hunyuan3-295b.yml` (full model) and `hunyuan3-smoke.yml` (same structure,
  4 layers).

## Switches

All default to off (upstream behaviour). Each has a unit test under `tests/unit/`.

| Switch | What it does |
|---|---|
| `routed_gate_bf16_rounding` | Router GateLogit: round the kernel to BF16 before the matmul (`reduce_precision`, so the gradient is not rounded) |
| `rmsnorm_single_rounding` | RMSNorm: normalize * scale in FP32, round to dtype once |
| `mlp_activation_round_per_op` | Gated MLP: each op in FP32, output rounded to dtype; the routed sparse path honours `activations_in_float32` |
| `dense_fp32_accumulation` | DenseGeneral (attention q/k/v/out, dense MLP): FP32 accumulation, rounded to dtype once |
| `rmsnorm_scale_grad_fp32` | RMSNorm scale gradient stays FP32 (with `rmsnorm_single_rounding`) |
| `logits_dense_wgrad_fp32` | Output head weight gradient stays FP32; forward unchanged |
| `embedding_grad_fp32` | Token-embedding gradient accumulates in FP32; forward unchanged |
| `remat_policy: save_everything` | Backward recomputes no forward op |
| `lm_head_dh_gauge_scale`, `lm_head_dh_wbar_remove` | Diagnostics only: perturb the hidden-state gradient coming from the logits |

Also included: ZeRO-1 optimizer-state resharding fixed for multi-process runs (`src/maxtext/utils/sharding.py`).
