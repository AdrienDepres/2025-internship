# test_rollout_md17.py

import numpy as np
import torch

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE = "estag_unimol_rollout_mais_progressif_best.pt"
NUM_PAST = 10
DELTA_FRAME = 10

START_FRAME = 7990


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
    in_edge_nf=NUM_PAST - 1, # att fft = True
    hidden_nf=32,
    fft=True,
    eat=True,
    device=DEVICE,
    n_layers=2,
    n_nodes=n_nodes,
    nodes_att_dim=NUM_PAST - 1, # a rajouter si fft = True ?
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

        pred = model(
            node_feat,
            current,
            edges,
            edge_attr,
        )

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

with open("test_du_dossier_estag_unimol", "w") as f:

    for frame in predicted_frames:

        if np.isnan(frame).any():
            break

        f.write(f"{len(Z)}\n")
        f.write("ESTAG + U rollout\n")

        for atom, xyz in zip(symbols, frame):

            x, y, z = xyz

            f.write(
                f"{atom:2s} "
                f"{x:.8f} "
                f"{y:.8f} "
                f"{z:.8f}\n"
            )

print("saved ->")