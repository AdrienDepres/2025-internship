import torch

checkpoint = torch.load(
    "estag_unimol_rollout_mais_progressif_best.pt",
    weights_only=True,
)

total_ckpt = sum(v.numel() for v in checkpoint.values())

print(f"Paramètres dans le checkpoint : {total_ckpt:,}")