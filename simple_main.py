# main_ani_train.py

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from ani1xmd.dataset import ANI1xDataset
from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG
import os.path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#NPZ_FILE = "code/MDtrajNet/data/ANI-1xMD/8/C2N2O4_681.npz"

NUM_PAST = 10
DELTA_FRAME = 10
BATCH_SIZE = 32

EPOCHS = 400
LR = 5e-4

# dataset = ANI1xDataset(
#     npz_file= NPZ_FILE,
#     max_samples=8000,
#     delta_frame=DELTA_FRAME,
#     num_past=NUM_PAST,
# )

ROLLOUT_STEPS = 1

dataset = MD17Dataset(
    partition="train",
    max_samples = 42000,
    data_dir="md17",
    molecule_type="aspirin",
    delta_frame=10,
    num_past=10,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataset = MD17Dataset(
    partition="test",
    max_samples=1000,
    data_dir="md17",
    molecule_type="aspirin",
    delta_frame=10,
    num_past=10,
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
    in_node_nf=513,
    in_edge_nf=NUM_PAST - 1,
    hidden_nf=32,
    fft=True,
    eat=True,
    device=DEVICE,
    n_layers=2,
    n_nodes=n_nodes,
    nodes_att_dim=NUM_PAST - 1,
).to(DEVICE)


pt_file = "resultats/estag_unimol_rollout_fft_aspirine.pt"
if os.path.isfile(pt_file):
    checkpoint = torch.load(
        pt_file,
        map_location=DEVICE,
        weights_only=True,      # PyTorch >= 2.0
    )

    model.load_state_dict(checkpoint)


optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
)

criterion = nn.MSELoss()

best_loss = [1e9] * 30

# if os.path.isfile("estag_unimol_best.pt"):
#     model.load_state_dict(
#         torch.load(
#             "estag_unimol_best.pt",
#             map_location=DEVICE,
#             weights_only=True
#         )
#     )










for epoch in range(314,EPOCHS):

    model.train()

    epoch_loss = 0.0

#     # for batch in loader:

#     #     loc, edge_attr, node_feat, loc_end = [
#     #         x.to(DEVICE)
#     #         for x in batch
#     #     ]

#     #     batch_size = loc.shape[0]

#     #     loc = (
#     #         loc.permute(1,0,2,3)
#     #            .reshape(loc.shape[1], -1, 3)
#     #     )

#     #     node_feat = node_feat.reshape(-1, 769)

#     #     loc_end = loc_end.reshape(-1, 3)

#     #     edges = dataset.get_edges(
#     #         batch_size,
#     #         n_nodes
#     #     )

#     #     edges = [e.to(DEVICE) for e in edges]

#     #     pred = model(
#     #         node_feat,
#     #         loc,
#     #         edges,
#     #         edge_attr.squeeze(0)
#     #     )

#     #     loss = criterion(
#     #         pred,
#     #         loc_end
#     #     )

#     #     optimizer.zero_grad()

#     #     loss.backward()

#     #     optimizer.step()

#     #     epoch_loss += loss.item()

#     # #epoch_loss /= len(loader)

#     # # print(
#     # #     f"epoch {epoch:4d} | loss = {epoch_loss:.8f}"
#     # # )

#     # -----------------------------------------
#     # TRAIN STEP : 10-step rollout loss
#     # -----------------------------------------

    

    ROLLOUT_STEPS = 1
    tc = 90
    # if epoch < tc + 10: # il faudrait peut-être augmenter la première partie à rollout = 1
    #     ROLLOUT_STEPS = 1
    # elif epoch < tc + 20:
    #     ROLLOUT_STEPS = 2
    # elif epoch < tc + 30:
    #     ROLLOUT_STEPS = 4
    # elif epoch < tc + 40:
    #     ROLLOUT_STEPS = 1
    # elif epoch < tc + 60:
    #     ROLLOUT_STEPS = 6
    # elif epoch < tc + 80:
    #     ROLLOUT_STEPS = 1
    # elif epoch < tc + 100:
    #     ROLLOUT_STEPS = 8
    # elif epoch < tc + 120:
    #     ROLLOUT_STEPS = 1
    # elif epoch < tc + 140:
    #     ROLLOUT_STEPS = 10
    # elif epoch < tc + 180:
    #     ROLLOUT_STEPS = 1
    # elif epoch < tc + 200:
    #     ROLLOUT_STEPS = 12
    # elif epoch < tc + 250:
    #     ROLLOUT_STEPS = 14
    # elif epoch < tc + 270:
    #     ROLLOUT_STEPS = 1
    # elif epoch < tc + 300:
    #     ROLLOUT_STEPS = 16
    # elif epoch < tc + 350:
    #     ROLLOUT_STEPS = 1
    # else:
    #     ROLLOUT_STEPS = 20
    
    if epoch <= tc + 0:
        ROLLOUT_STEPS = 1
    elif epoch <= tc + 20:
        ROLLOUT_STEPS = 2
    elif epoch <= tc + 40:
        ROLLOUT_STEPS = 4
    elif epoch <= tc + 60:
        ROLLOUT_STEPS = 6
    elif epoch <= tc + 80:
        ROLLOUT_STEPS = 8
    elif epoch <= tc + 100:
        ROLLOUT_STEPS = 10
    else:
        ROLLOUT_STEPS = NUM_PAST + 1

    # weights = torch.linspace(
    # 1.0,
    # 2.0,
    # ROLLOUT_STEPS,
    # device=DEVICE
    # )

    weights = torch.full(
    (ROLLOUT_STEPS,),
    1.0 / ROLLOUT_STEPS,
    device=DEVICE
)

#     for batch in loader:

#         loc, edge_attr, node_feat, _ , start_idx = [
#             x.to(DEVICE)
#             for x in batch
#         ]

#         start_idx = start_idx.item()

#         if start_idx + DELTA_FRAME * (NUM_PAST + ROLLOUT_STEPS) >= len(dataset.x):
#                 continue

#         batch_size = loc.shape[0]

#         loc = (
#             loc.permute(1,0,2,3)
#             .reshape(loc.shape[1], -1, 3)
#         )

#         node_feat = node_feat.reshape(-1, 513)

#         edges = dataset.get_edges(
#             batch_size,
#             n_nodes
#         )
#         edges = [e.to(DEVICE) for e in edges]

#         optimizer.zero_grad()

#         current_window = loc.clone()

#         rollout_loss = 0.0

#         # frame correspondant à x[t]
#         #start_idx = batch_idx

#         for k in range(ROLLOUT_STEPS):

#             pred = model(
#                 node_feat,
#                 current_window,
#                 edges,
#                 edge_attr.squeeze(0)
#             )

#             target_frame = torch.tensor(
#                 dataset.x[
#                     start_idx
#                     + DELTA_FRAME * (NUM_PAST + k + 1)
#                 ],
#                 dtype=torch.float32,
#                 device=DEVICE
#             )

#             # rollout_loss += criterion(
#             #     pred,
#             #     target_frame
#             # )

#             rollout_loss += weights[k] * criterion(
#                 pred,
#                 target_frame
#             )

#             current_window = torch.cat(
#                 [
#                     current_window[1:],
#                     pred.unsqueeze(0)
#                 ],
#                 dim=0
#             )

#         # rollout_loss /= ROLLOUT_STEPS MERDE, ça cause la baisse de la loss pour aucune raison

#         rollout_loss.backward()

#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(),
#             1.0
#         )

#         optimizer.step()

#         epoch_loss += rollout_loss.item()

#     epoch_loss /= len(loader)

    # -----------------------------------------
    # TRAIN STEP : batched rollout loss
    # -----------------------------------------

    model.train()

    epoch_loss = 0.0
    n_batches = 0

    for batch in loader:

        loc, edge_attr, node_feat, _, start_idx = [
            x.to(DEVICE)
            for x in batch
        ]

        # start_idx : (B,)
        start_idx = start_idx.long()

        # On enlève les samples qui n'ont pas assez de futur
        last_needed_idx = start_idx + DELTA_FRAME * (
            NUM_PAST + ROLLOUT_STEPS - 1
        )

        valid_mask = last_needed_idx < len(dataset.x)

        if not valid_mask.any():
            continue

        loc = loc[valid_mask]
        edge_attr = edge_attr[valid_mask]
        node_feat = node_feat[valid_mask]
        start_idx = start_idx[valid_mask]

        batch_size = loc.shape[0]

        # loc : (B, T, N, 3)
        # -> current_window : (T, B*N, 3)
        current_window = (
            loc.permute(1, 0, 2, 3)
            .reshape(NUM_PAST, batch_size * n_nodes, 3)
        )

        node_feat = node_feat.reshape(
            batch_size * n_nodes,
            -1,
        )

        # edge_attr : (B, E, 3)
        # Important : pour batch_size > 1, surtout pas squeeze(0)
        edge_attr_batch = edge_attr.reshape(
            -1,
            edge_attr.shape[-1],
        )

        edges = dataset.get_edges(
            batch_size,
            n_nodes,
        )
        edges = [e.to(DEVICE) for e in edges]

        optimizer.zero_grad()

        rollout_loss = 0.0

        for k in range(ROLLOUT_STEPS):

            pred = model(
                node_feat,
                current_window,
                edges,
                edge_attr_batch,
            )

            # Target batchée
            # k=0 correspond à la vraie frame :
            # start + DELTA_FRAME * NUM_PAST
            target_indices = (
                start_idx
                + DELTA_FRAME * (NUM_PAST + k)
            )

            target_np = dataset.x[
                target_indices.detach().cpu().numpy()
            ]

            target_frame = torch.as_tensor(
                target_np,
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(batch_size * n_nodes, 3)

            rollout_loss = rollout_loss + weights[k] * criterion(
                pred,
                target_frame,
            )

            # Très important :
            # on injecte la version corrigée dans le rollout,
            # sinon la correction stabilise la loss mais pas la dynamique.
            current_window = torch.cat(
                [
                    current_window[1:],
                    pred.unsqueeze(0),
                ],
                dim=0,
            )

        rollout_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        epoch_loss += rollout_loss.item()
        n_batches += 1

    if n_batches > 0:
        epoch_loss /= n_batches



    model.eval()

    val_loss = 0.0
    n_val_batches = 0

    # # with torch.no_grad():

    # #     for batch in val_loader:

    # #         loc, edge_attr, node_feat, loc_end = [
    # #             x.to(DEVICE)
    # #             for x in batch
    # #         ]

    # #         batch_size = loc.shape[0]

    # #         loc = (
    # #             loc.permute(1,0,2,3)
    # #             .reshape(loc.shape[1], -1, 3)
    # #         )

    # #         node_feat = node_feat.reshape(-1, 769)
    # #         loc_end = loc_end.reshape(-1, 3)

    # #         edges = val_dataset.get_edges(
    # #             batch_size,
    # #             val_dataset.n_node
    # #         )
    # #         edges = [e.to(DEVICE) for e in edges]

    # #         pred = model(
    # #             node_feat,
    # #             loc,
    # #             edges,
    # #             edge_attr.squeeze(0)
    # #         )

    # #         loss = criterion(pred, loc_end)

    # #         val_loss += loss.item()
    # #         n_val_batches += 1

    # #val_loss /= n_val_batches

    # with torch.no_grad():
    #     for batch in val_loader:

    #         loc, edge_attr, node_feat, _ , start_idx = [
    #             x.to(DEVICE)
    #             for x in batch
    #         ]

    #         start_idx = start_idx.item()

    #         if start_idx + DELTA_FRAME * (NUM_PAST + ROLLOUT_STEPS) >= len(val_dataset.x):
    #             continue

    #         batch_size = loc.shape[0]

    #         loc = (
    #             loc.permute(1,0,2,3)
    #             .reshape(loc.shape[1], -1, 3)
    #         )

    #         node_feat = node_feat.reshape(-1, 513)

    #         edges = val_dataset.get_edges(
    #             batch_size,
    #             n_nodes
    #         )
    #         edges = [e.to(DEVICE) for e in edges]

    #         current_window = loc.clone()

    #         val_rollout_loss = 0.0

    #         # frame correspondant à x[t]
    #         #start_idx = batch_idx

    #         for k in range(ROLLOUT_STEPS):

    #             pred = model(
    #                 node_feat,
    #                 current_window,
    #                 edges,
    #                 edge_attr.squeeze(0)
    #             )

    #             target_frame = torch.tensor(
    #                 val_dataset.x[
    #                     start_idx
    #                     + DELTA_FRAME * (NUM_PAST + k + 1)
    #                 ],
    #                 dtype=torch.float32,
    #                 device=DEVICE
    #             )

    #             # val_rollout_loss += criterion(
    #             #     pred,
    #             #     target_frame
    #             # )

    #             val_rollout_loss += weights[k] * criterion(
    #                 pred,
    #                 target_frame
    #             )

    #             current_window = torch.cat(
    #                 [
    #                     current_window[1:],
    #                     pred.unsqueeze(0)
    #                 ],
    #                 dim=0
    #             )
            
    #         val_loss += (val_rollout_loss.item() / ROLLOUT_STEPS)

    #     val_loss /= len(val_loader)



    with torch.no_grad():
        for batch in val_loader:

            loc, edge_attr, node_feat, _, start_idx = [
                x.to(DEVICE)
                for x in batch
            ]

            # start_idx : (B,)
            start_idx = start_idx.long()

            # On enlève les samples qui n'ont pas assez de futur
            last_needed_idx = start_idx + DELTA_FRAME * (
                NUM_PAST + ROLLOUT_STEPS - 1
            )

            valid_mask = last_needed_idx < len(val_dataset.x)

            if not valid_mask.any():
                continue

            loc = loc[valid_mask]
            edge_attr = edge_attr[valid_mask]
            node_feat = node_feat[valid_mask]
            start_idx = start_idx[valid_mask]

            batch_size = loc.shape[0]

            # loc : (B, T, N, 3)
            # -> current_window : (T, B*N, 3)
            current_window = (
                loc.permute(1, 0, 2, 3)
                .reshape(NUM_PAST, batch_size * n_nodes, 3)
            )

            node_feat = node_feat.reshape(
                batch_size * n_nodes,
                -1,
            )

            # edge_attr : (B, E, 3)
            # Important : pour batch_size > 1, surtout pas squeeze(0)
            edge_attr_batch = edge_attr.reshape(
                -1,
                edge_attr.shape[-1],
            )

            edges = dataset.get_edges(
                batch_size,
                n_nodes,
            )
            edges = [e.to(DEVICE) for e in edges]


            rollout_loss = 0.0

            for k in range(1):

                pred = model(
                    node_feat,
                    current_window,
                    edges,
                    edge_attr_batch,
                )

                # Target batchée
                # k=0 correspond à la vraie frame :
                # start + DELTA_FRAME * NUM_PAST
                target_indices = (
                    start_idx
                    + DELTA_FRAME * (NUM_PAST + k)
                )

                target_np = val_dataset.x[
                    target_indices.detach().cpu().numpy()
                ]

                target_frame = torch.as_tensor(
                    target_np,
                    dtype=torch.float32,
                    device=DEVICE,
                ).reshape(batch_size * n_nodes, 3)

                rollout_loss = rollout_loss + 1 * criterion(
                    pred,
                    target_frame,
                )

                # Très important :
                # on injecte la version corrigée dans le rollout,
                # sinon la correction stabilise la loss mais pas la dynamique.
                current_window = torch.cat(
                    [
                        current_window[1:],
                        pred.unsqueeze(0),
                    ],
                    dim=0,
                )


            val_loss += rollout_loss.item()
            n_val_batches += 1

        if n_val_batches > 0:
            val_loss /= n_val_batches


    print(
        f"epoch {epoch:4d} | "
        f"train = {epoch_loss:.8f} | "
        f"val = {val_loss:.8f} |"
        f"rollout_steps = {ROLLOUT_STEPS}"
    )

    # if epoch == 60:
    #     best_loss = 1e9
    
    # if epoch == 80:
    #     best_loss = 1e9

    if val_loss < best_loss[ROLLOUT_STEPS]: 

        best_loss[ROLLOUT_STEPS] = val_loss

        torch.save(
            model.state_dict(),
            pt_file
        )

        with open("logs/entrainements_logs/best_model_log_aspirine_modele_classique_EUR_fft_oskour.txt", "a") as record_log:
            record_log.write(
                f"epoch={epoch} "
                f"train_loss={epoch_loss:.8e} "
                f"val_loss={val_loss:.8e}\n"
            )

        print(
            f"saved model (loss={best_loss[ROLLOUT_STEPS]:.8f})"
        )

print()
print("training finished")
print("best loss =", best_loss)
