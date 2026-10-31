# main_ani_train.py

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from ani1xmd.dataset import ANI1xDataset
from models.model_unimol import ESTAG
import os.path
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ_FILE = "code/MDtrajNet/data/ANI-1xMD/8/C2N2O4_681.npz"

NUM_PAST = 10
DELTA_FRAME = 10
BATCH_SIZE = 1

EPOCHS = 100
LR = 5e-4

dataset = ANI1xDataset(
    npz_file= NPZ_FILE,
    max_samples=8000,
    delta_frame=DELTA_FRAME,
    num_past=NUM_PAST,
    rollout=True,
    rollout_steps = 5
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataset = ANI1xDataset(
    npz_file=NPZ_FILE,
    max_samples=1000,
    delta_frame=DELTA_FRAME,
    num_past=NUM_PAST,
    validation=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

n_nodes = dataset.n_node

model = ESTAG(
    num_past=NUM_PAST,
    num_future=1,
    in_node_nf=769,
    in_edge_nf=3,
    hidden_nf=32,
    fft=False,
    eat=True,
    device=DEVICE,
    n_layers=2,
    n_nodes=n_nodes,
).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5,
)

criterion = nn.MSELoss()

best_loss = 1e9

if os.path.isfile("estag_unimol_best.pt"):
    model.load_state_dict(
        torch.load(
            "estag_unimol_best.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )


for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0.0

    for batch in loader:

        loc, edge_attr, node_feat, loc_future = [
            x.to(DEVICE)
            for x in batch
        ]
        



        batch_size = loc.shape[0]

        loc = (
            loc.permute(1,0,2,3)
            .reshape(loc.shape[1], -1, 3)
        )

        node_feat = node_feat.reshape(-1, 769)

        loc_future = (
            loc_future.permute(1,0,2,3)
                    .reshape(loc_future.shape[1],
                            -1,
                            3)
        )

        edges = dataset.get_edges(
            batch_size,
            n_nodes
        )

        edges = [e.to(DEVICE) for e in edges]

        optimizer.zero_grad()

        history = loc.clone()

        rollout_loss = 0.0

        K = loc_future.shape[0]

        for k in range(K):

            pred = model(
                node_feat,
                history,
                edges,
                edge_attr.squeeze(0)
            )

            target = loc_future[k]

            rollout_loss += criterion(
                pred,
                target
            )

            history = torch.cat(
                [
                    history[1:], # Couteux de fou ça non ?
                    pred.unsqueeze(0)
                ],
                dim=0
            )

        rollout_loss = rollout_loss / K

        rollout_loss.backward()

        optimizer.step()

        epoch_loss += rollout_loss.item()

    #-------------------------------------------------------------------------------------------------------------
    print("début de validation")
    model.eval()

    val_loss = 0.0
    n_val_batches = 0

    with torch.no_grad():

        for batch in val_loader:

            loc, edge_attr, node_feat, loc_end = [
                x.to(DEVICE)
                for x in batch
            ]
            


            batch_size = loc.shape[0]

            loc = (
                loc.permute(1,0,2,3)
                .reshape(loc.shape[1], -1, 3)
            )

            node_feat = node_feat.reshape(-1, 769)
            loc_end = loc_end.reshape(-1, 3)

            edges = val_dataset.get_edges(
                batch_size,
                val_dataset.n_node
            )
            edges = [e.to(DEVICE) for e in edges]

            pred = model(
                node_feat,
                loc,
                edges,
                edge_attr.squeeze(0)
            )

            loss = criterion(pred, loc_end)

            val_loss += loss.item()
            n_val_batches += 1

    #val_loss /= n_val_batches

    print(
        f"epoch {epoch:4d} | "
        f"train = {epoch_loss:.8f} | "
        f"val = {val_loss:.8f}"
    )

    if val_loss < best_loss: 

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "estag_unimol_best_rollout.pt"
        )

        print(
            f"saved model (loss={best_loss:.8f})"
        )

print()
print("training finished")
print("best loss =", best_loss)