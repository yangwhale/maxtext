# Branch `hunyuan4`: Hunyuan 4 on MaxText

This branch is upstream [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext) plus the changes we
used to run Hunyuan 4 (Hy4, the architecture of the public
[Hy4-preview](https://huggingface.co/tencent/Hy4-preview) checkpoint) on TPU and to align its numerics with the GPU
(Megatron-Core) reference. It is published as a single commit on top of the upstream commit it was developed
against, so `git diff HEAD~1` shows the whole change set.

The numerical alignment work that drove these changes (method, golden runs, per-level checks, the full switch
matrix and which switches each run uses) is in [yangwhale/janus](https://github.com/yangwhale/janus), under
`models/hunyuan4/`. In the janus documents, "the Hunyuan 4 integration branch of maxtext" means this branch.

## Model

- Built on the DeepSeek decoder (`src/maxtext/models/deepseek.py`): MLA with an elementwise output gate, DSA indexer
  with IndexCache (`indexer_share_interval`), hyper-connections (mHC / iHC, optional Pallas kernels in
  `src/maxtext/kernels/ihc/`), HC head, MTP, sigmoid routing with expert bias.
- Configs in `src/maxtext/configs/models/`: `hy4-preview.yml` (full model), `hy4-mid.yml` (~5.19B), `hy4-mini.yml`
  (~0.43B) and `hy4-tiny.yml` (unit tests), all with the same checkpoint reading.
- HF -> MaxText checkpoint conversion for Hy4-preview, de-interleaving the q_pe / k_pe columns at conversion
  (`checkpoint_rope_interleave`).

## Switches

All default to off (upstream behaviour) unless noted. Most have a unit test under `tests/unit/`.

Precision of activations, weights and gradients:

| Switch | What it does |
|---|---|
| `rmsnorm_single_rounding`, `rmsnorm_scale_grad_fp32`, `rmsnorm_scale_grad_gpu_order` | RMSNorm rounds once; scale gradient in FP32, or summed in the TE / Megatron order |
| `mlp_activation_round_per_op`, `mlp_fused_fc1_dgrad` | Gated MLP rounding per op; SwiGLU up-projection input gradient rounded once, as a fused fc1 |
| `dense_fp32_accumulation` | DenseGeneral FP32 accumulation, rounded once |
| `logits_dense_wgrad_fp32`, `logits_matmul_precision` | Output head: weight gradient in FP32; its own matmul precision |
| `embedding_grad_fp32` | Token-embedding gradient accumulates in FP32 (correct under data parallelism + gradient accumulation) |
| `embedding_grad_fp32_local_bf16_comm` | FP32 local scatter-add, rounded to BF16 once, BF16 data-parallel all-reduce |
| `embedding_shard_over_data` | Shard only the embedding table's hidden dim over `data`, so pure data parallelism partitions it like FSDP |
| `router_matmul_precision`, `routed_gate_bf16_rounding`, `moe_router_grad_from_wo_input`, `moe_mcore_prob_before_fc2` | Router precision and Megatron-Core ordering of the routing weight |
| `float32_qk_product`, `float32_logits` | Now also honoured by MLA (passed through to AttentionOp) |
| `splash_dsinks_fp32`, `splash_dsinks_gpu_order` | Attention-sink gradient in FP32, or in the Megatron-Core order |
| `transcendental_accuracy_highest`, `transcendental_log_freebsd`, `accurate_sigmoid_*`, `accurate_silu_*` | Accurate FP32 exp / log / sigmoid / silu, per call site |
| `ihc_force_bf16_rounding`, `ihc_stream_sum_unrolled`, `hc_matmul_precision` | Hyper-connection rounding and precision |

Megatron-Core compatibility (optimizer, losses, indexer):

| Switch | What it does |
|---|---|
| `muon_mcore_nesterov`, `muon_adam_plain`, `muon_adam_accurate`, `muon_mcore_ns_normalize`, `muon_mcore_ns_bf16`, `muon_exclude_ihc`, `muon_fuse_glu`, `muon_attention_gate_whole_matrix` | Muon details as Megatron-Core / emerging_optimizers |
| `qk_clip_mcore_denominator`, `qk_clip_max_logit_exclude_sink`, `qk_clip_log_per_head` | QK-Clip details and logging |
| `indexer_fp32_scores`, `indexer_weights_proj_bf16_weight`, `indexer_kd_mcore_weight`, `indexer_kd_mcore_log`, `indexer_kd_generalized_kl` | DSA indexer scoring and KD loss |
| `mtp_loss_mcore_per_rank`, `mtp_input_post_final_norm` | MTP loss weighting and input |

Memory and speed: `remat_policy: save_everything`, `moe_out` / `moe_expert_weights` / `attention_gate` remat
choices, `use_ihc_pallas_kernel`, `gmm_v2_native_transpose_rhs`, `indexer_kth_select`,
`splash_static_causal_partial_mask`, `moe_dropless_retry`.

Diagnostics (no numerical effect): `debug_forced_routing_deepseek`, `debug_forced_indexer_topk`,
`debug_capture_choices`, `debug_routing_digest`, `debug_sow_indexer`, `log_indexer_loss_per_layer`.
