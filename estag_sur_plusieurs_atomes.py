import glob
import random
import os
from ani1xmd.dataset import ANI1xDataset
import torch
from torch import nn, optim
from models.model_unimol import ESTAG

random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_PAST = 10
DELTA_FRAME = 10
#BATCH_SIZE = 1

EPOCHS = 200
LR = 5e-4

all_files = sorted(
    glob.glob("code/MDtrajNet/data/ANI-1xMD/8/*.npz")
)

random.shuffle(all_files)

n = len(all_files)

train_files = all_files[:int(0.8*n)]
val_seen_files = train_files[:5]
val_unseen_files = all_files[n-5:]


train_sets = [
    ANI1xDataset(
        npz_file=f,
        #name=os.path.basename(f),
        max_samples=8000,
        delta_frame=10,
        num_past=10,
    )
    for f in train_files[:42]
]

val_seen_sets = [
    ANI1xDataset(
        npz_file=f,
        #name=os.path.basename(f),
        validation=True,
        max_samples=1000,
        delta_frame=10,
        num_past=10,
    )
    for f in val_seen_files[:42]
]

val_unseen_sets = [
    ANI1xDataset(
        npz_file=f,
        #name=os.path.basename(f),
        validation=True,
        max_samples=1000,
        delta_frame=10,
        num_past=10,
    )
    for f in val_unseen_files
]

print(len(train_sets))
print(len(val_seen_sets))
print(len(val_unseen_sets))

from torch.utils.data import ConcatDataset, DataLoader

# train_dataset = ConcatDataset(train_sets)

# val_seen_dataset = ConcatDataset(val_seen_sets)

# val_unseen_dataset = ConcatDataset(val_unseen_sets)




# train_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True,
# )

# val_seen_loader = DataLoader(
#     val_seen_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=4,
# )

# val_unseen_loader = DataLoader(
#     val_unseen_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=4,
# )

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



optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
)

criterion = nn.MSELoss()

# best_loss = 1e9


# for epoch in range(EPOCHS):

#     model.train()

#     epoch_loss = 0.0

#     for batch in train_loader:

#         loc, edge_attr, node_feat, loc_end = [
#             x.to(DEVICE)
#             for x in batch
#         ]

#         batch_size = loc.shape[0]

#         loc = (
#             loc.permute(1,0,2,3)
#                .reshape(loc.shape[1], -1, 3)
#         )

#         node_feat = node_feat.reshape(-1, 769)

#         loc_end = loc_end.reshape(-1, 3)

#         edges = dataset.get_edges(
#             batch_size,
#             n_nodes
#         )

#         edges = [e.to(DEVICE) for e in edges]

#         pred = model(
#             node_feat,
#             loc,
#             edges,
#             edge_attr.squeeze(0)
#         )

#         loss = criterion(
#             pred,
#             loc_end
#         )

#         optimizer.zero_grad()

#         loss.backward()

#         optimizer.step()

#         epoch_loss += loss.item()

#     epoch_loss /= len(train_loader)

#     # print(
#     #     f"epoch {epoch:4d} | loss = {epoch_loss:.8f}"
#     # )


#     model.eval()

#     val_loss = 0.0
#     n_val_batches = 0

#     with torch.no_grad():

#         for batch in val_seen_loader:

#             loc, edge_attr, node_feat, loc_end = [
#                 x.to(DEVICE)
#                 for x in batch
#             ]

#             batch_size = loc.shape[0]

#             loc = (
#                 loc.permute(1,0,2,3)
#                 .reshape(loc.shape[1], -1, 3)
#             )

#             node_feat = node_feat.reshape(-1, 769)
#             loc_end = loc_end.reshape(-1, 3)

#             edges = val_dataset.get_edges(
#                 batch_size,
#                 val_dataset.n_node
#             )
#             edges = [e.to(DEVICE) for e in edges]

#             pred = model(
#                 node_feat,
#                 loc,
#                 edges,
#                 edge_attr.squeeze(0)
#             )

#             loss = criterion(pred, loc_end)

#             val_loss += loss.item()
#             n_val_batches += 1

#             # mol_name = name[0]

#             # losses.setdefault(mol_name, [])

#             # losses[mol_name].append(loss.item())

#     val_loss /= (n_val_batches * batch_size)


    

#     val_unseen_loss = 0.0
#     n_val_batches = 0

#     with torch.no_grad():

#         for batch in val_unseen_loader:

#             loc, edge_attr, node_feat, loc_end = [
#                 x.to(DEVICE)
#                 for x in batch
#             ]

#             batch_size = loc.shape[0]

#             loc = (
#                 loc.permute(1,0,2,3)
#                 .reshape(loc.shape[1], -1, 3)
#             )

#             node_feat = node_feat.reshape(-1, 769)
#             loc_end = loc_end.reshape(-1, 3)

#             edges = val_dataset.get_edges(
#                 batch_size,
#                 val_dataset.n_node
#             )
#             edges = [e.to(DEVICE) for e in edges]

#             pred = model(
#                 node_feat,
#                 loc,
#                 edges,
#                 edge_attr.squeeze(0)
#             )

#             loss = criterion(pred, loc_end)

#             val_unseen_loss += loss.item()
#             n_val_batches += 1

#     val_unseen_loss /= (n_val_batches * batch_size)

#     print(
#         f"epoch {epoch:4d} | "
#         f"train = {epoch_loss:.8f} | "
#         f"val_seen_loss = {val_loss:.8f} |"
#         f"val_unseen_loss = {val_unseen_loss:.8f}"
#     )

#     if val_unseen_loss < best_loss:

#         best_loss = val_unseen_loss

#         torch.save(
#             model.state_dict(),
#             "estag_unimol_general.pt"
#         )

#         with open("best_model_log_04_07.txt", "a") as record_log:
#             record_log.write(
#                 f"epoch={epoch} "
#                 f"train_loss={epoch_loss:.8e} "
#                 f"val_seen_loss={val_loss:.8e}"
#                 f"val_unseen_loss={val_unseen_loss:.8e}\n"
#             )
#         print("saved model !")




best_loss = 1e9

for epoch in range(EPOCHS):

    ############################################################
    # TRAIN
    ############################################################

    model.train()
    epoch_loss = 0.0
    n_train_batches = 0

    random.shuffle(train_sets)

    for dataset in train_sets[:42]:

        train_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )

        n_nodes = dataset.n_node

        for batch in train_loader:

            loc, edge_attr, node_feat, loc_end = [
                x.to(DEVICE)
                for x in batch
            ]

            batch_size = loc.shape[0]

            loc = (
                loc.permute(1, 0, 2, 3)
                   .reshape(loc.shape[1], -1, 3)
            )

            node_feat = node_feat.reshape(-1, 769)
            loc_end = loc_end.reshape(-1, 3)

            edges = dataset.get_edges(
                batch_size,
                n_nodes
            )
            edges = [e.to(DEVICE) for e in edges]

            edge_attr = edge_attr.reshape(-1, 3)

            pred = model(
                node_feat,
                loc,
                edges,
                edge_attr
            )

            loss = criterion(
                pred,
                loc_end
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            n_train_batches += 1

    epoch_loss /= n_train_batches


    ############################################################
    # VALIDATION : molécules déjà vues
    ############################################################

    model.eval()

    val_seen_loss = 0.0
    n_seen_batches = 0

    # with torch.no_grad():

    #     for dataset in val_seen_sets:

    #         loader = DataLoader(
    #             dataset,
    #             batch_size=32,
    #             shuffle=False,
    #             num_workers=0,
    #         )

    #         n_nodes = dataset.n_node

    #         for batch in loader:

    #             loc, edge_attr, node_feat, loc_end = [
    #                 x.to(DEVICE)
    #                 for x in batch
    #             ]

    #             batch_size = loc.shape[0]

    #             loc = (
    #                 loc.permute(1,0,2,3)
    #                    .reshape(loc.shape[1], -1, 3)
    #             )

    #             node_feat = node_feat.reshape(-1,769)
    #             loc_end = loc_end.reshape(-1,3)

    #             edges = dataset.get_edges(
    #                 batch_size,
    #                 n_nodes
    #             )
    #             edges = [e.to(DEVICE) for e in edges]

    #             edge_attr = edge_attr.reshape(-1,3)

    #             pred = model(
    #                 node_feat,
    #                 loc,
    #                 edges,
    #                 edge_attr
    #             )

    #             loss = criterion(
    #                 pred,
    #                 loc_end
    #             )

    #             val_seen_loss += loss.item()
    #             n_seen_batches += 1

    # val_seen_loss /= n_seen_batches


    ############################################################
    # VALIDATION : molécules jamais vues
    ############################################################

    val_unseen_loss = 0.0
    n_unseen_batches = 0

    with torch.no_grad():

        for dataset in val_unseen_sets:

            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
            )

            n_nodes = dataset.n_node

            for batch in loader:

                loc, edge_attr, node_feat, loc_end = [
                    x.to(DEVICE)
                    for x in batch
                ]

                batch_size = loc.shape[0]

                loc = (
                    loc.permute(1,0,2,3)
                       .reshape(loc.shape[1], -1, 3)
                )

                node_feat = node_feat.reshape(-1,769)
                loc_end = loc_end.reshape(-1,3)

                edges = dataset.get_edges(
                    batch_size,
                    n_nodes
                )
                edges = [e.to(DEVICE) for e in edges]

                edge_attr = edge_attr.reshape(-1,3)

                pred = model(
                    node_feat,
                    loc,
                    edges,
                    edge_attr
                )

                loss = criterion(
                    pred,
                    loc_end
                )

                val_unseen_loss += loss.item()
                n_unseen_batches += 1

    val_unseen_loss /= n_unseen_batches


    ############################################################
    # LOG
    ############################################################

    print(
        f"epoch {epoch:4d} | "
        f"train={epoch_loss:.8f} | "
        #f"seen={val_seen_loss:.8f} | "
        f"unseen={val_unseen_loss:.8f}"
    )


    ############################################################
    # SAVE
    ############################################################

    if val_unseen_loss < best_loss:

        best_loss = val_unseen_loss

        torch.save(
            model.state_dict(),
            "estag_unimol_general.pt"
        )

        with open("best_model_log_04_07.txt", "a") as f:

            f.write(
                f"epoch={epoch} "
                f"train={epoch_loss:.8e} "
                f"seen={val_seen_loss:.8e} "
                f"unseen={val_unseen_loss:.8e}\n"
            )

        print("saved model !")

print()
print("Training finished")
print("Best validation loss :", best_loss)