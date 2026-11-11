# Generated with: Megatron-Core 685fb8b (core 0.20.0), emerging_optimizers 0.3.0, torch 2.11.0 CPU.
# PYTHONPATH=<Megatron-LM checkout> python gen_ref.py   (writes muon_mcore_reference.npz)
"""Megatron-Core 0.20 TensorParallelMuon (emerging_optimizers 0.3.0) + torch Adam reference updates for the MaxText
muon_mcore_* / muon_adam_plain switches. CPU FP32, highest precision. torch layout [out, in]."""
import os, numpy as np, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", str(29700 + os.getpid() % 100))
torch.distributed.init_process_group("gloo", rank=0, world_size=1)
from megatron.core import parallel_state as ps; ps.initialize_model_parallel(1, 1)
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
torch.set_float32_matmul_precision("highest")
rng = np.random.default_rng(0)
shapes = {"wide": (48, 16), "tall": (16, 40), "square": (24, 24)}  # [out, in]
T = 5
res = {}
for k, s in shapes.items():
    G = rng.standard_normal((T,) + s).astype(np.float32) * 0.01
    G[:, :, 0] *= 1e-5  # one tiny column, so the normalization matters a little
    res[f"{k}_grads"] = G
    p = torch.nn.Parameter(torch.zeros(s))
    opt = TensorParallelMuon([p], lr=1.0, momentum=0.95, nesterov=True, weight_decay=0.0, split_qkv=False,
        is_qkv_fn=lambda q: False, qkv_split_shapes=[1], coefficient_type="simple", num_ns_steps=5,
        scale_mode="shape_scaling", fp32_matmul_prec="highest")
    ups = []
    for t in range(T):
        before = p.detach().clone(); p.grad = torch.from_numpy(G[t].copy()); opt.step(); ups.append((p.detach() - before).numpy())
    res[f"{k}_updates"] = np.stack(ups)
G = rng.standard_normal((T, 32)).astype(np.float32) * 0.01
res["scalar_grads"] = G
p = torch.nn.Parameter(torch.zeros(32)); opt = torch.optim.Adam([p], lr=1.0, betas=(0.9, 0.95), eps=1e-10)
ups = []
for t in range(T):
    before = p.detach().clone(); p.grad = torch.from_numpy(G[t].copy()); opt.step(); ups.append((p.detach() - before).numpy())
res["scalar_updates"] = np.stack(ups)
np.savez_compressed("muon_mcore_reference.npz", **res)
print({k: v.shape for k, v in res.items()})
