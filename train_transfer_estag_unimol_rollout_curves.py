# train_transfer_estag_unimol_rollout_curves.py
#
# 1) Entraine ESTAG + UniMol + rollout sur aspirin.
# 2) Transfere les couches compatibles vers une autre molecule, par defaut uracil.
# 3) Evalue regulierement des rollouts classiques et stocke les MSE aux steps 1, 11, 100, 200.
# 4) Trace les courbes a partir des CSV.
#
# Hypothese : MD17Dataset.__getitem__ renvoie :
#     loc, edge_attr, node_feat, loc_end, start_idx

import os
import csv
import copy
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# CONFIG
# ============================================================

NUM_PAST = 10
DELTA_FRAME = 10

IN_NODE_NF = 513
IN_EDGE_NF = 3
HIDDEN_NF = 32
N_LAYERS = 2

BATCH_SIZE = 32
EVAL_ROLLOUT_BATCH_SIZE = 8

LR = 5e-4
WEIGHT_DECAY = 1e-4

SOURCE_TRAIN_SAMPLES = 80000
TARGET_TRAIN_SAMPLES = 30000
VAL_SAMPLES = 2000
EVAL_ROLLOUT_SAMPLES = 100

EVAL_EVERY = 10
MEASURE_STEPS = [1, 11, 100, 200]
MAX_ROLLOUT_STEP = max(MEASURE_STEPS)

SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(molecule, partition, max_samples, data_dir):
    return MD17Dataset(
        partition=partition,
        max_samples=max_samples,
        data_dir=data_dir,
        molecule_type=molecule,
        delta_frame=DELTA_FRAME,
        num_past=NUM_PAST,
    )


def make_rollout_subset(dataset, n_samples):
    n = min(n_samples, len(dataset))
    return Subset(dataset, list(range(n)))


def make_model(n_nodes):
    return ESTAG(
        num_past=NUM_PAST,
        num_future=1,
        in_node_nf=IN_NODE_NF,
        in_edge_nf=NUM_PAST - 1,
        hidden_nf=HIDDEN_NF,
        fft=True,
        eat=True,
        device=DEVICE,
        n_layers=N_LAYERS,
        n_nodes=n_nodes,
        nodes_att_dim=NUM_PAST - 1,
    ).to(DEVICE)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def transfer_compatible_weights(source_state, target_model):
    """Charge tous les poids ayant meme nom et meme shape."""

    target_state = target_model.state_dict()
    loaded = {}
    skipped = []

    for name, weight in source_state.items():
        if name not in target_state:
            skipped.append((name, "absent target"))
            continue
        if target_state[name].shape != weight.shape:
            skipped.append((name, f"{tuple(weight.shape)} -> {tuple(target_state[name].shape)}"))
            continue
        loaded[name] = weight

    target_state.update(loaded)
    target_model.load_state_dict(target_state)
    return len(loaded), skipped


def unwrap_dataset(dataset_or_subset):
    if isinstance(dataset_or_subset, Subset):
        return dataset_or_subset.dataset
    return dataset_or_subset


def unpack_batch(batch):
    if len(batch) != 5:
        raise ValueError(
            "Le batch doit contenir 5 elements : loc, edge_attr, node_feat, loc_end, start_idx. "
            "Modifie MD17Dataset.__getitem__ pour renvoyer self.st[i]."
        )

    loc, edge_attr, node_feat, loc_end, start_idx = batch
    return (
        loc.to(DEVICE),
        edge_attr.to(DEVICE),
        node_feat.to(DEVICE),
        loc_end.to(DEVICE),
        start_idx.to(DEVICE).long(),
    )


def valid_start_mask(base_dataset, start_idx, rollout_steps):
    last_needed_idx = start_idx + DELTA_FRAME * (NUM_PAST + rollout_steps - 1)
    return last_needed_idx < len(base_dataset.x)


def prepare_tensors(loc, edge_attr, node_feat, loc_end, start_idx, base_dataset):
    batch_size = loc.shape[0]
    n_nodes = base_dataset.n_node

    loc = loc.permute(1, 0, 2, 3).reshape(NUM_PAST, batch_size * n_nodes, 3)
    edge_attr_batch = edge_attr.reshape(-1, edge_attr.shape[-1])
    node_feat = node_feat.reshape(batch_size * n_nodes, -1)
    loc_end = loc_end.reshape(batch_size * n_nodes, 3)

    edges = base_dataset.get_edges(batch_size, n_nodes)
    edges = [e.to(DEVICE) for e in edges]

    return loc, edge_attr_batch, node_feat, loc_end, start_idx, edges, batch_size


def get_ground_truth(base_dataset, target_indices):
    idx = target_indices.detach().cpu().numpy().astype(int)
    target_np = base_dataset.x[idx]
    return torch.as_tensor(target_np, dtype=torch.float32, device=DEVICE)


def get_train_rollout_steps(epoch):
    """Progressivite rollout : 1 -> 2 -> 4 -> 6 -> 8 -> 10 -> 11."""

    epoch -= 100
    if epoch <= 0:
        return 1
    if epoch <= 20:
        return 2
    if epoch <= 40:
        return 4
    if epoch <= 60:
        return 6
    if epoch <= 80:
        return 8
    if epoch <= 100:
        return 10
    return NUM_PAST + 1


def train_one_epoch(model, optimizer, criterion, loader, dataset_or_subset, rollout_steps):
    model.train()
    base_dataset = unwrap_dataset(dataset_or_subset)
    n_nodes = base_dataset.n_node

    weights = torch.full((rollout_steps,), 1.0 / rollout_steps, dtype=torch.float32, device=DEVICE)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        loc, edge_attr, node_feat, loc_end, start_idx = unpack_batch(batch)

        mask = valid_start_mask(base_dataset, start_idx, rollout_steps)
        if not mask.any():
            continue

        loc = loc[mask]
        edge_attr = edge_attr[mask]
        node_feat = node_feat[mask]
        loc_end = loc_end[mask]
        start_idx = start_idx[mask]

        current_window, edge_attr_batch, node_feat, loc_end, start_idx, edges, batch_size = prepare_tensors(
            loc, edge_attr, node_feat, loc_end, start_idx, base_dataset
        )

        optimizer.zero_grad()
        rollout_loss = 0.0

        for k in range(rollout_steps):
            pred = model(node_feat, current_window, edges, edge_attr_batch)

            target_indices = start_idx + DELTA_FRAME * (NUM_PAST + k)
            target = get_ground_truth(base_dataset, target_indices).reshape(batch_size * n_nodes, 3)

            rollout_loss = rollout_loss + weights[k] * criterion(pred, target)
            current_window = torch.cat([current_window[1:], pred.unsqueeze(0)], dim=0)

        if not torch.isfinite(rollout_loss):
            print("WARNING: non-finite train loss, batch skipped")
            continue

        rollout_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += rollout_loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


@torch.no_grad()
def evaluate_one_step(model, criterion, loader, dataset_or_subset):
    model.eval()
    base_dataset = unwrap_dataset(dataset_or_subset)

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        loc, edge_attr, node_feat, loc_end, start_idx = unpack_batch(batch)
        loc, edge_attr_batch, node_feat, loc_end, start_idx, edges, batch_size = prepare_tensors(
            loc, edge_attr, node_feat, loc_end, start_idx, base_dataset
        )

        pred = model(node_feat, loc, edges, edge_attr_batch)
        loss = criterion(pred, loc_end)

        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


@torch.no_grad()
def evaluate_rollout_classic(model, loader, dataset_or_subset, measure_steps):
    """Rollout classique : pas de recentrage, pas de Kabsch."""

    model.eval()
    base_dataset = unwrap_dataset(dataset_or_subset)
    n_nodes = base_dataset.n_node
    max_step = max(measure_steps)

    mse_values = {step: [] for step in measure_steps}
    n_nan = {step: 0 for step in measure_steps}
    n_valid_samples = 0
    n_skipped_samples = 0

    for batch in loader:
        loc, edge_attr, node_feat, loc_end, start_idx = unpack_batch(batch)

        mask = valid_start_mask(base_dataset, start_idx, max_step)
        if not mask.any():
            n_skipped_samples += start_idx.numel()
            continue

        loc = loc[mask]
        edge_attr = edge_attr[mask]
        node_feat = node_feat[mask]
        loc_end = loc_end[mask]
        start_idx = start_idx[mask]

        current_window, edge_attr_batch, node_feat, loc_end, start_idx, edges, batch_size = prepare_tensors(
            loc, edge_attr, node_feat, loc_end, start_idx, base_dataset
        )

        n_valid_samples += batch_size

        for step in range(1, max_step + 1):
            pred = model(node_feat, current_window, edges, edge_attr_batch)

            target_indices = start_idx + DELTA_FRAME * (NUM_PAST + step - 1)
            target = get_ground_truth(base_dataset, target_indices).reshape(batch_size * n_nodes, 3)

            if step in measure_steps:
                per_sample_mse = (pred - target).reshape(batch_size, n_nodes, 3).pow(2).mean(dim=(1, 2))
                finite = torch.isfinite(per_sample_mse)

                if finite.any():
                    mse_values[step].extend(per_sample_mse[finite].detach().cpu().tolist())

                n_nan[step] += int((~finite).sum().item())

            current_window = torch.cat([current_window[1:], pred.unsqueeze(0)], dim=0)

    result = {}
    for step in measure_steps:
        vals = np.asarray(mse_values[step], dtype=np.float64)
        result[step] = {
            "mean_mse": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
            "n_finite": int(len(vals)),
            "n_nan": int(n_nan[step]),
        }

    result["n_valid_samples"] = int(n_valid_samples)
    result["n_skipped_samples"] = int(n_skipped_samples)
    return result


def init_csv_files(out_dir):
    epoch_csv = os.path.join(out_dir, "epoch_metrics.csv")
    rollout_csv = os.path.join(out_dir, "rollout_metrics.csv")
    transfer_csv = os.path.join(out_dir, "transfer_summary.csv")

    if not os.path.isfile(epoch_csv):
        with open(epoch_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stage", "molecule", "epoch", "global_epoch", "train_rollout_steps",
                "train_loss", "val_one_step_mse", "checkpoint",
            ])

    if not os.path.isfile(rollout_csv):
        with open(rollout_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "stage", "molecule", "epoch", "global_epoch",
                "mse_step_1", "mse_step_11", "mse_step_100", "mse_step_200",
                "n_finite_step_1", "n_finite_step_11", "n_finite_step_100", "n_finite_step_200",
                "n_nan_step_1", "n_nan_step_11", "n_nan_step_100", "n_nan_step_200",
                "n_valid_samples", "n_skipped_samples",
            ])

    if not os.path.isfile(transfer_csv):
        with open(transfer_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["from_stage", "to_stage", "from_molecule", "to_molecule", "n_loaded", "n_skipped"])

    return epoch_csv, rollout_csv, transfer_csv


def append_epoch_metrics(epoch_csv, stage, molecule, epoch, global_epoch, rollout_steps, train_loss, val_mse, checkpoint):
    with open(epoch_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([stage, molecule, epoch, global_epoch, rollout_steps, train_loss, val_mse, checkpoint])


def append_rollout_metrics(rollout_csv, stage, molecule, epoch, global_epoch, result):
    with open(rollout_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            stage, molecule, epoch, global_epoch,
            result[1]["mean_mse"], result[11]["mean_mse"], result[100]["mean_mse"], result[200]["mean_mse"],
            result[1]["n_finite"], result[11]["n_finite"], result[100]["n_finite"], result[200]["n_finite"],
            result[1]["n_nan"], result[11]["n_nan"], result[100]["n_nan"], result[200]["n_nan"],
            result["n_valid_samples"], result["n_skipped_samples"],
        ])


def append_transfer_summary(transfer_csv, from_stage, to_stage, from_molecule, to_molecule, n_loaded, n_skipped):
    with open(transfer_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([from_stage, to_stage, from_molecule, to_molecule, n_loaded, n_skipped])


def safe_load_csv(path):
    if not os.path.isfile(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        return np.atleast_1d(data)
    except Exception:
        return None


def plot_curves(out_dir):
    epoch_csv = os.path.join(out_dir, "epoch_metrics.csv")
    rollout_csv = os.path.join(out_dir, "rollout_metrics.csv")

    epoch_data = safe_load_csv(epoch_csv)
    rollout_data = safe_load_csv(rollout_csv)

    if epoch_data is not None and epoch_data.size > 0:
        x = np.asarray(epoch_data["global_epoch"], dtype=float)
        train_loss = np.asarray(epoch_data["train_loss"], dtype=float)
        val_loss = np.asarray(epoch_data["val_one_step_mse"], dtype=float)

        plt.figure()
        plt.plot(x, train_loss, label="train rollout loss")
        plt.plot(x, val_loss, label="val one-step MSE")
        plt.xlabel("Global epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_train_val_loss.png"), dpi=200)
        plt.close()

    if rollout_data is not None and rollout_data.size > 0:
        x = np.asarray(rollout_data["global_epoch"], dtype=float)

        plt.figure()
        for step in MEASURE_STEPS:
            y = np.asarray(rollout_data[f"mse_step_{step}"], dtype=float)
            plt.plot(x, y, marker="o", label=f"step {step}")

            plt.figure()
            plt.plot(x, y, marker="o")
            plt.xlabel("Global epoch")
            plt.ylabel(f"MSE rollout step {step}")
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"curve_rollout_mse_step_{step}.png"), dpi=200)
            plt.close()

        plt.xlabel("Global epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_rollout_mse_all_steps.png"), dpi=200)
        plt.close()


def train_stage(
    stage, molecule, epochs, train_samples, data_dir, out_dir,
    previous_state, previous_stage, previous_molecule, global_epoch_start,
    epoch_csv, rollout_csv, transfer_csv,
):
    print("\n" + "=" * 80)
    print(f"STAGE {stage} | molecule = {molecule}")
    print("=" * 80)

    stage_dir = os.path.join(out_dir, f"stage_{stage:02d}_{molecule}")
    os.makedirs(stage_dir, exist_ok=True)

    train_dataset = make_dataset(molecule, "train", train_samples, data_dir)
    val_dataset = make_dataset(molecule, "test", VAL_SAMPLES, data_dir)
    rollout_eval_base = make_dataset(molecule, "test", max(VAL_SAMPLES, EVAL_ROLLOUT_SAMPLES), data_dir)
    rollout_eval_dataset = make_rollout_subset(rollout_eval_base, EVAL_ROLLOUT_SAMPLES)

    n_nodes = train_dataset.n_node
    print("n_nodes         :", n_nodes)
    print("train samples   :", len(train_dataset))
    print("val samples     :", len(val_dataset))
    print("rollout samples :", len(rollout_eval_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    rollout_loader = DataLoader(
        rollout_eval_dataset,
        batch_size=EVAL_ROLLOUT_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = make_model(n_nodes)
    print("params          :", count_parameters(model))

    if previous_state is not None:
        n_loaded, skipped = transfer_compatible_weights(previous_state, model)
        print("transferred     :", n_loaded)
        print("skipped         :", len(skipped))

        with open(os.path.join(stage_dir, "skipped_weights.txt"), "w") as f:
            for name, reason in skipped:
                f.write(f"{name} | {reason}\n")

        append_transfer_summary(
            transfer_csv,
            previous_stage,
            stage,
            previous_molecule,
            molecule,
            n_loaded,
            len(skipped),
        )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_checkpoint = ""
    global_epoch = global_epoch_start

    for epoch in range(1, epochs + 1):
        global_epoch += 1
        rollout_steps = get_train_rollout_steps(epoch)

        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, train_dataset, rollout_steps)
        val_mse = evaluate_one_step(model, criterion, val_loader, val_dataset)

        checkpoint = ""
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = os.path.join(stage_dir, "best.pt")
            torch.save(best_state, checkpoint)
            best_checkpoint = checkpoint

        append_epoch_metrics(
            epoch_csv, stage, molecule, epoch, global_epoch,
            rollout_steps, train_loss, val_mse, checkpoint,
        )

        do_rollout_eval = epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs

        if do_rollout_eval:
            rollout_result = evaluate_rollout_classic(model, rollout_loader, rollout_eval_dataset, MEASURE_STEPS)
            append_rollout_metrics(rollout_csv, stage, molecule, epoch, global_epoch, rollout_result)

            print(
                f"{molecule:10s} | epoch {epoch:4d}/{epochs} | glob {global_epoch:4d} | "
                f"train={train_loss:.8e} | val1={val_mse:.8e} | "
                f"r1={rollout_result[1]['mean_mse']:.8e} | "
                f"r11={rollout_result[11]['mean_mse']:.8e} | "
                f"r100={rollout_result[100]['mean_mse']:.8e} | "
                f"r200={rollout_result[200]['mean_mse']:.8e}"
            )
            plot_curves(out_dir)

        else:
            print(
                f"{molecule:10s} | epoch {epoch:4d}/{epochs} | glob {global_epoch:4d} | "
                f"train={train_loss:.8e} | val1={val_mse:.8e} | rollout_train={rollout_steps}"
            )

    last_checkpoint = os.path.join(stage_dir, "last.pt")
    torch.save(model.state_dict(), last_checkpoint)

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_checkpoint = last_checkpoint

    print("best checkpoint:", best_checkpoint)
    print("last checkpoint:", last_checkpoint)

    return best_state, stage, molecule, global_epoch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="md17")
    parser.add_argument("--out_dir", type=str, default="resultats/transfer_estag_unimol_rollout")

    parser.add_argument("--source_mol", type=str, default="aspirin")
    parser.add_argument("--target_mol", type=str, default="uracil")

    parser.add_argument("--source_epochs", type=int, default=150)
    parser.add_argument("--target_epochs", type=int, default=150)

    parser.add_argument("--source_train_samples", type=int, default=SOURCE_TRAIN_SAMPLES)
    parser.add_argument("--target_train_samples", type=int, default=TARGET_TRAIN_SAMPLES)

    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    epoch_csv, rollout_csv, transfer_csv = init_csv_files(args.out_dir)

    print("DEVICE          :", DEVICE)
    print("OUT DIR         :", args.out_dir)
    print("SOURCE          :", args.source_mol)
    print("TARGET          :", args.target_mol)
    print("MEASURE STEPS   :", MEASURE_STEPS)
    print("EVAL EVERY      :", EVAL_EVERY)
    print("EVAL ROLLOUT N  :", EVAL_ROLLOUT_SAMPLES)

    previous_state = None
    previous_stage = None
    previous_molecule = None
    global_epoch = 0

    previous_state, previous_stage, previous_molecule, global_epoch = train_stage(
        stage=1,
        molecule=args.source_mol,
        epochs=args.source_epochs,
        train_samples=args.source_train_samples,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        previous_state=previous_state,
        previous_stage=previous_stage,
        previous_molecule=previous_molecule,
        global_epoch_start=global_epoch,
        epoch_csv=epoch_csv,
        rollout_csv=rollout_csv,
        transfer_csv=transfer_csv,
    )

    previous_state, previous_stage, previous_molecule, global_epoch = train_stage(
        stage=2,
        molecule=args.target_mol,
        epochs=args.target_epochs,
        train_samples=args.target_train_samples,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        previous_state=previous_state,
        previous_stage=previous_stage,
        previous_molecule=previous_molecule,
        global_epoch_start=global_epoch,
        epoch_csv=epoch_csv,
        rollout_csv=rollout_csv,
        transfer_csv=transfer_csv,
    )

    plot_curves(args.out_dir)

    print("\nTermine.")
    print("CSV epoch   :", epoch_csv)
    print("CSV rollout :", rollout_csv)
    print("CSV transfer:", transfer_csv)
    print("Figures     :", args.out_dir)


if __name__ == "__main__":
    main()
