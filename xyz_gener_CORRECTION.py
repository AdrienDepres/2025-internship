# test_rollout_md17.py

import numpy as np
import torch

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE = "estag_unimol_rollout_prog_corrige.pt"
NUM_PAST = 10
DELTA_FRAME = 10

START_FRAME = 7990


def kabsch_align_torch_single(mobile, reference):
    """
    mobile, reference : (N, 3)

    Aligne mobile sur reference par translation + rotation optimale.
    """

    mobile_center = mobile.mean(dim=0, keepdim=True)
    reference_center = reference.mean(dim=0, keepdim=True)

    X = mobile - mobile_center
    Y = reference - reference_center

    C = X.T @ Y

    U, S, Vh = torch.linalg.svd(C)

    D = torch.eye(
        3,
        device=mobile.device,
        dtype=mobile.dtype,
    )

    D[-1, -1] = torch.sign(
        torch.det(U @ Vh)
    )

    R = U @ D @ Vh

    mobile_aligned = X @ R + reference_center

    return mobile_aligned


def correct_frame_if_needed(
    pred,
    reference,
    com_threshold=0.00001,
    mode="kabsch",
):
    """
    pred      : frame prédite, (N,3)
    reference : frame précédente, (N,3)

    Corrige seulement si le centre géométrique dérive trop.

    mode="kabsch"      : translation + rotation optimale
    mode="translation" : translation seulement
    """

    pred_com = pred.mean(dim=0)
    ref_com = reference.mean(dim=0)

    drift = torch.linalg.norm(
        pred_com - ref_com
    )

    if drift <= com_threshold:
        return pred, False, drift.item()

    if mode == "translation":
        pred_corrected = pred - pred_com + ref_com

    elif mode == "kabsch":
        pred_corrected = kabsch_align_torch_single(
            pred,
            reference,
        )

    else:
        raise ValueError(f"Unknown correction mode: {mode}")

    return pred_corrected, True, drift.item()




# ============================================================
# DATASET
# ============================================================

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

print(
    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=DEVICE,
            weights_only=True,
        ),
        strict=True,
    )
)

model.eval()


# ============================================================
# NODE FEATURES
# ============================================================

# node_feat = (
#     dataset.mole_idx
#     .unsqueeze(-1)
#     .float()
#     .to(DEVICE)
# )
node_feat = dataset[0][2]
node_feat = node_feat.float().to(DEVICE)

print("node_feat :", node_feat.shape)


# ============================================================
# GRAPH
# ============================================================

edges = dataset.get_edges(
    batch_size=1,
    n_nodes=n_nodes,
)

edges = [
    edges[0].to(DEVICE),
    edges[1].to(DEVICE),
]

edge_attr = dataset.edge_attr.to(DEVICE)

print("edge_attr :", edge_attr.shape)


# ============================================================
# TRAJECTORY
# ============================================================

all_xyz = dataset.x

start = 108000 # TODO clarifier start / START_FRAME

window = torch.tensor(
    all_xyz[
        start - DELTA_FRAME*(NUM_PAST-1) :
        start + 1 :
        DELTA_FRAME
    ],
    dtype=torch.float32,
    device=DEVICE
)

window = window.reshape(
    NUM_PAST,
    n_nodes,
    3,
)

print("window :", window.shape)


# ============================================================
# ROLLOUT
# ============================================================

predicted_frames = []

with torch.no_grad():

    current = window.clone()
    c = 0
    while True:

        pred_raw = model(
            node_feat,
            current,
            edges,
            edge_attr,
        )

        reference_frame = current[-1]

        pred, was_corrected, drift = correct_frame_if_needed(
            pred=pred_raw,
            reference=reference_frame,
            com_threshold=0.00000001,   # à ajuster
            mode="kabsch",      # ou "translation"
        )

        # if was_corrected:
        #     print(
        #         f"correction à frame {len(predicted_frames)} | "
        #         f"drift COM = {drift:.4f} Å"
        #     )

        predicted_frames.append(
            pred.cpu().numpy()
        )

        current = torch.cat(
            [
                current[1:],
                pred.unsqueeze(0),
            ],
            dim=0,
        )

        frame_id = (
            start
            + (len(predicted_frames) + 1) * DELTA_FRAME
        )

        if frame_id >= len(all_xyz):
            break
        c+=1
        if c > 2000:
            break


predicted_frames = np.asarray(
    predicted_frames
)

print(
    "generated",
    predicted_frames.shape[0],
    "frames"
)


# ============================================================
# ATOMIC SYMBOLS
# ============================================================

data = np.load(
    "md17/aspirin_dft.npz"
)

Z = data["z"]

periodic_table = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}

symbols = [
    periodic_table[int(z)]
    for z in Z
]


# ============================================================
# XYZ OUTPUT
# ============================================================

with open("E_U_R_W_corrige_pour_de_vrai.xyz", "w") as f:

    for frame in predicted_frames:

        if np.isnan(frame).any():
            break

        f.write(f"{len(Z)}\n")
        f.write("ESTAG rollout\n")

        for atom, xyz in zip(symbols, frame):

            x, y, z = xyz

            f.write(
                f"{atom:2s} "
                f"{x:.8f} "
                f"{y:.8f} "
                f"{z:.8f}\n"
            )

print("saved ->")