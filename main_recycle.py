# simple_main_unimol_like_main_md.py

import argparse
import os
import json
import random

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import matplotlib.pyplot as plt

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="ESTAG + UniMol, proche de main_md.py")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--no_cuda", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--test_interval", type=int, default=5)

parser.add_argument("--outf", type=str, default="logs/md17_unimol_logs")
parser.add_argument("--exp_name", type=str, default="resultats/estag_unimol/")

parser.add_argument("--nf", type=int, default=32)
parser.add_argument("--n_layers", type=int, default=2)

parser.add_argument("--max_training_samples", type=int, default=42000)
parser.add_argument("--weight_decay", type=float, default=1e-12)
parser.add_argument("--lr", type=float, default=5e-4)

parser.add_argument("--data_dir", type=str, default="md17")
parser.add_argument("--mol", type=str, default="aspirin")

parser.add_argument("--num_past", type=int, default=10)
parser.add_argument("--delta_frame", type=int, default=10)

parser.add_argument("--fft", type=eval, default=True)
parser.add_argument("--eat", type=eval, default=True)
parser.add_argument("--with_mask", action="store_true", default=False)
parser.add_argument("--tempo", type=eval, default=True)
parser.add_argument("--save_m", type=eval, default=True)

# Dimension de node_feat renvoyé par ton dataset.py.
# Si node_feat = [Z + UniMol1], et UniMol1 vaut 512, alors 513.
parser.add_argument("--in_node_nf", type=int, default=513)

args = parser.parse_args()


# ============================================================
# SETUP
# ============================================================

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

loss_mse = nn.MSELoss()

print(args)
print("device:", device)

os.makedirs(args.outf, exist_ok=True)

exp_path = os.path.join(
    args.outf,
    f"exp_{args.mol}",
    args.exp_name,
)

os.makedirs(exp_path, exist_ok=True)


# ============================================================
# OUTILS
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def unpack_dataset_output(data):
    """
    Ton dataset.py a été modifié.

    Cas acceptés :
        4 éléments : loc, edge_attr, node_feat, loc_end
        5 éléments : loc, edge_attr, node_feat, loc_end, start_idx

    start_idx est ignoré ici, car on reste en entraînement one-step
    comme main_md.py.
    """

    if len(data) == 4:
        loc, edge_attr, node_feat, loc_end = data

    elif len(data) == 5:
        loc, edge_attr, node_feat, loc_end, _start_idx = data

    else:
        raise ValueError(
            f"Dataset output inattendu : {len(data)} éléments. "
            "Attendu : 4 ou 5."
        )

    return loc, edge_attr, node_feat, loc_end


def reshape_like_main_md(data):
    """
    Reproduit l'esprit de main_md.py, mais sans appliquer aveuglément
    la même transformation à tout.

    Entrées batchées :
        loc       : (B, T, N, 3)
        edge_attr : (B, E, Fe) ou (E, Fe) selon ton dataset/collate
        node_feat : (B, N, F)
        loc_end   : (B, N, 3)

    Sorties :
        loc       : (T, B*N, 3)
        edge_attr : (B*E, Fe)
        node_feat : (B*N, F)
        loc_end   : (B*N, 3)
    """

    loc, edge_attr, node_feat, loc_end = data

    batch_size = loc.shape[0]
    num_past = loc.shape[1]
    n_nodes = loc.shape[2]

    # loc : (B,T,N,3) -> (T,B*N,3)
    loc = (
        loc.permute(1, 0, 2, 3)
           .reshape(num_past, batch_size * n_nodes, 3)
    )

    # node_feat : (B,N,F) -> (B*N,F)
    node_feat = node_feat.reshape(
        batch_size * n_nodes,
        node_feat.shape[-1],
    )

    # loc_end : (B,N,3) -> (B*N,3)
    loc_end = loc_end.reshape(
        batch_size * n_nodes,
        3,
    )

    # edge_attr :
    # Si DataLoader reçoit self.edge_attr identique pour chaque sample,
    # il produit en général (B,E,Fe).
    # Pour get_edges(batch_size, n_nodes), il faut donc aussi B copies :
    # (B,E,Fe) -> (B*E,Fe).
    if edge_attr.dim() == 3:
        edge_attr = edge_attr.reshape(
            -1,
            edge_attr.shape[-1],
        )

    elif edge_attr.dim() == 2:
        # Cas rare : déjà (E,Fe), seulement valable si batch_size=1
        if batch_size != 1:
            raise ValueError(
                "edge_attr a shape (E,Fe) mais batch_size > 1. "
                "Il faut que le dataset renvoie edge_attr par sample, "
                "afin que le DataLoader produise (B,E,Fe)."
            )

    else:
        raise ValueError(
            f"edge_attr.dim() inattendu : {edge_attr.dim()}"
        )

    return loc, edge_attr, node_feat, loc_end, batch_size, n_nodes


def check_node_features(node_feat, expected_dim):

    if node_feat.shape[-1] != expected_dim:
        raise ValueError(
            f"node_feat a dimension {node_feat.shape[-1]}, "
            f"mais args.in_node_nf={expected_dim}. "
            "Si UniMol1 vaut 512 et que tu concatènes Z, il faut 513."
        )


# ============================================================
# TRAIN / EVAL
# ============================================================

def train(model, optimizer, epoch, loader, backprop=True):

    if backprop:
        model.train()
    else:
        model.eval()

    res = {
        "epoch": epoch,
        "loss": 0.0,
        "counter": 0,
        "reg_loss": 0.0,
    }

    context = torch.enable_grad() if backprop else torch.no_grad()

    with context:

        for batch_idx, raw_data in enumerate(loader):

            raw_data = [
                d.to(device)
                for d in raw_data
                if torch.is_tensor(d)
            ]

            loc, edge_attr, node_feat, loc_end = unpack_dataset_output(raw_data)

            check_node_features(
                node_feat,
                args.in_node_nf,
            )

            loc, edge_attr, node_feat, loc_end, batch_size, n_nodes = reshape_like_main_md(
                (loc, edge_attr, node_feat, loc_end)
            )

            edges = loader.dataset.get_edges(
                batch_size,
                n_nodes,
            )

            edges = [
                edges[0].to(device),
                edges[1].to(device),
            ]

            if backprop:
                optimizer.zero_grad()

            loc_pred = model(
                node_feat,
                loc,
                edges,
                edge_attr,
            )

            # Même logique que main_md.py :
            # loc_pred : (B*N,3)
            # loc_end  : (B*N,3)
            loss = loss_mse(
                loc_pred,
                loc_end,
            )

            if backprop:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

            # Même pondération que main_md.py :
            # on moyenne par sample/molécule, pas par batch.
            res["loss"] += loss.item() * batch_size
            res["counter"] += batch_size

    prefix = "" if backprop else "==> "

    avg_loss = res["loss"] / res["counter"]

    print(
        "%s%s epoch %d avg loss: %.8f reg loss: %.8f"
        % (
            prefix,
            loader.dataset.partition,
            epoch,
            avg_loss,
            res["reg_loss"] / max(res["counter"], 1),
        )
    )

    return avg_loss


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(args.seed)

    dataset_train = MD17Dataset(
        partition="train",
        max_samples=args.max_training_samples,
        data_dir=args.data_dir,
        molecule_type=args.mol,
        delta_frame=args.delta_frame,
        num_past=args.num_past,
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    dataset_val = MD17Dataset(
        partition="val",
        max_samples=2000,
        data_dir=args.data_dir,
        molecule_type=args.mol,
        delta_frame=args.delta_frame,
        num_past=args.num_past,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    dataset_test = MD17Dataset(
        partition="test",
        max_samples=2000,
        data_dir=args.data_dir,
        molecule_type=args.mol,
        delta_frame=args.delta_frame,
        num_past=args.num_past,
    )

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    in_edge_nf = dataset_train.edge_attr.shape[-1]
    nodes_att_dim = 0

    if args.fft:
        in_edge_nf = args.num_past - 1
        nodes_att_dim = args.num_past - 1

    n_nodes = dataset_train.x_t.shape[1]

    print("n_nodes:", n_nodes)
    print("in_edge_nf:", in_edge_nf)
    print("in_node_nf:", args.in_node_nf)

    model = ESTAG(
        num_past=args.num_past,
        num_future=1,
        in_node_nf=args.in_node_nf,
        in_edge_nf=in_edge_nf,
        hidden_nf=args.nf,
        fft=args.fft,
        eat=args.eat,
        nodes_att_dim=nodes_att_dim,
        device=device,
        n_layers=args.n_layers,
        n_nodes=n_nodes,
        with_mask=args.with_mask,
        tempo=args.tempo,
    ).to(device)

    print(model)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("parameters:", n_params)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    results = {
        "epochs": [],
        "test loss": [],
        "val loss": [],
        "train loss": [],
    }

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_train_loss = 1e8
    best_epoch = 0

    for epoch in range(args.epochs):

        train_loss = train(
            model,
            optimizer,
            epoch,
            loader_train,
            backprop=True,
        )

        results["train loss"].append(train_loss)

        if epoch % args.test_interval == 0:

            val_loss = train(
                model,
                optimizer,
                epoch,
                loader_val,
                backprop=False,
            )

            test_loss = train(
                model,
                optimizer,
                epoch,
                loader_test,
                backprop=False,
            )

            results["epochs"].append(epoch)
            results["val loss"].append(val_loss)
            results["test loss"].append(test_loss)

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch

                if args.save_m:

                    torch.save(
                        model.state_dict(),
                        os.path.join(exp_path, "saved_model_state_dict.pt"),
                    )

                    torch.save(
                        model,
                        os.path.join(exp_path, "saved_model_full.pth"),
                    )

            print(
                "*** Best Val Loss: %.8f \t Best Test Loss: %.8f \t Best epoch %d"
                % (
                    best_val_loss,
                    best_test_loss,
                    best_epoch,
                )
            )

        with open(os.path.join(exp_path, "loss.json"), "w") as outfile:

            json.dump(
                results,
                outfile,
                indent=4,
            )

    return best_train_loss, best_val_loss, best_test_loss, best_epoch


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()

    print("best_train = %.8f" % best_train_loss)
    print("best_val   = %.8f" % best_val_loss)
    print("best_test  = %.8f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)

    with open(os.path.join(exp_path, "loss.json")) as f:
        loss = json.load(f)

    plt.figure()

    if len(loss["train loss"]) >= 5:
        train_smooth = [
            np.mean(loss["train loss"][i * 5 : (i + 1) * 5])
            for i in range(len(loss["train loss"]) // 5)
        ]

        plt.plot(
            loss["epochs"][: len(train_smooth)],
            train_smooth,
            label="Train",
        )

    plt.plot(
        loss["epochs"],
        loss["val loss"],
        label="Val",
    )

    plt.plot(
        loss["epochs"],
        loss["test loss"],
        label="Test",
    )

    plt.legend()
    plt.title("Loss")
    plt.tight_layout()

    plt.savefig(
        os.path.join(exp_path, "loss.png")
    )