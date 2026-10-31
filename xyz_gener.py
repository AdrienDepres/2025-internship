# test_rollout.py

import numpy as np
import torch

from ani1xmd.dataset import ANI1xDataset
from models.model_unimol import ESTAG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_PAST = 10
DELTA_FRAME = 10

NPZ_FILE = "code/MDtrajNet/data/ANI-1xMD/8/C2N2O4_681.npz"
MODEL_FILE = "estag_unimol_general.pt"

dataset = ANI1xDataset(
    npz_file=NPZ_FILE,
    max_samples=8000,
    delta_frame=DELTA_FRAME,
    num_past=NUM_PAST,
)

n_nodes = dataset.n_node

model = ESTAG(
    num_past=10,
    num_future=1,
    in_node_nf=769,
    in_edge_nf=3,
    hidden_nf=32,
    fft=False,
    eat=True,
    device=DEVICE,
    n_layers=2,
    n_nodes=8,
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

node_feat = dataset.node_feat.to(DEVICE)

edges = dataset.get_edges(
    batch_size=1,
    n_nodes=n_nodes
)

edges = [e.to(DEVICE) for e in edges]

edge_attr = dataset.edge_attr.to(DEVICE)

all_xyz = dataset.xyz
start_frame = 7990

window = torch.tensor(
    all_xyz[start_frame:start_frame+NUM_PAST],
    dtype=torch.float32,
    device=DEVICE
)

predicted_frames = []

with torch.no_grad():

    current = window.clone()

    while True:

        pred = model(
            node_feat,
            current,
            edges,
            edge_attr
        )

        predicted_frames.append(
            pred.cpu().numpy()
        )

        current = torch.cat(
            [
                current[1:],
                pred.unsqueeze(0)
            ],
            dim=0
        )

        frame_id = (
            start_frame
            + NUM_PAST
            + len(predicted_frames)
        )

        if frame_id >= len(all_xyz):
            break

predicted_frames = np.asarray(predicted_frames)

print(
    "generated",
    predicted_frames.shape[0],
    "frames"
)


data = np.load(NPZ_FILE)

symbols = data["sp"][0]

with open("rollout_unimol_plusieurs_mol.xyz", "w") as f:

    for frame in predicted_frames:

        f.write(f"{len(symbols)}\n")
        f.write("ESTAG rollout\n")

        for atom, xyz in zip(symbols, frame):

            x, y, z = xyz

            f.write(
                f"{atom:2s} "
                f"{x:.8f} "
                f"{y:.8f} "
                f"{z:.8f}\n"
            )