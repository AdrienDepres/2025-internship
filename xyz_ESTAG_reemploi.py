import numpy as np
import torch

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE = "estag_unimol_rollout_mais_progressif_best.pt"
NUM_PAST = 10
DELTA_FRAME = 10

START_FRAME = 7990



def load_compatible_weights(model, checkpoint_path, device, exclude_theta=False):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_state = model.state_dict()

    compatible = {}

    for name, weight in checkpoint.items():

        if exclude_theta and "theta" in name:
            continue

        if name not in model_state:
            print(f"[skip] {name} absent du nouveau modèle")
            continue

        if model_state[name].shape != weight.shape:
            print(
                f"[skip] {name} shape mismatch: "
                f"{weight.shape} -> {model_state[name].shape}"
            )
            continue

        compatible[name] = weight

    model_state.update(compatible)
    model.load_state_dict(model_state)

    print(f"{len(compatible)} couches chargées.")







dataset = MD17Dataset(
    partition="test",
    max_samples=2000,
    data_dir="md17",
    molecule_type="aspirin",
    delta_frame=DELTA_FRAME,
    num_past=NUM_PAST,
)

n_nodes = dataset.n_node

print("n_nodes =", n_nodes)


# ============================================================
# MODEL
# ============================================================

model = ESTAG(
    num_past=NUM_PAST,
    num_future=1,
    in_node_nf=513,
    in_edge_nf=3,
    hidden_nf=32,
    fft=False,
    eat=True,
    device=DEVICE,
    n_layers=2,
    n_nodes=n_nodes,
).to(DEVICE)



load_compatible_weights(
    model,
    "estag_unimol_rollout_mais_progressif_best.pt",
    DEVICE,
    exclude_theta=True,
)