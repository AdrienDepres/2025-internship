# train_estag_unimol_multimol.py

import os
import csv
import copy
import torch
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "resultats/estag_plusieurs_mol"
LOG_FILE = os.path.join(RESULT_DIR, "metrics.csv")

os.makedirs(RESULT_DIR, exist_ok=True)

DATA_DIR = "md17"

MOLECULE_SEQUENCE = [
    "aspirin",
    "uracil",
    "salicylic",
    "malonaldehyde",
    "aspirin",

]

NUM_PAST = 10
DELTA_FRAME = 10

IN_NODE_NF = 513
IN_EDGE_NF = 3
HIDDEN_NF = 32
N_LAYERS = 2

BATCH_SIZE = 32
EPOCHS_PER_MOL = 100

LR = 5e-4
WEIGHT_DECAY = 1e-4

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 1000

# Important :
# Avec NUM_PAST=10, la 10e prédiction utilise encore
# une frame originale dans la fenêtre.
# La 11e est la première prédiction avec une fenêtre 100% prédite.
TRAIN_ROLLOUT_STEPS = NUM_PAST + 1

EVAL_ROLLOUT_STEPS = [11, 100, 200, 1000]

DIVERGENCE_MSE_THRESHOLD = 0.1
DIVERGENCE_MAX_STEP = 2000

EVAL_EVERY = 10

# Les rollouts longs coûtent cher.
# Ici on ne les évalue qu'à la dernière époque de chaque molécule.
ROLLOUT_EVAL_ONLY_LAST_EPOCH = True

SAVE_ROLLOUT_XYZ = True
MAX_SAVED_ROLLOUT_TRAJS = 5


PTABLE = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
}


# ============================================================
# DATASETS / MODELE
# ============================================================

def make_dataset(molecule, partition, max_samples):
    return MD17Dataset(
        partition=partition,
        max_samples=max_samples,
        data_dir=DATA_DIR,
        molecule_type=molecule,
        delta_frame=DELTA_FRAME,
        num_past=NUM_PAST,
    )


def make_model(n_nodes):
    return ESTAG(
        num_past=NUM_PAST,
        num_future=1,
        in_node_nf=IN_NODE_NF,
        in_edge_nf=IN_EDGE_NF,
        hidden_nf=HIDDEN_NF,
        fft=False,
        eat=True,
        device=DEVICE,
        n_layers=N_LAYERS,
        n_nodes=n_nodes,
    ).to(DEVICE)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def transfer_compatible_weights(source_state, target_model):
    """
    Charge le maximum de poids compatibles :
    même nom + même shape.
    """

    target_state = target_model.state_dict()

    loaded = {}
    skipped = []

    for name, weight in source_state.items():

        if name not in target_state:
            skipped.append((name, "absent du nouveau modèle"))
            continue

        if target_state[name].shape != weight.shape:
            skipped.append(
                (
                    name,
                    f"{tuple(weight.shape)} -> {tuple(target_state[name].shape)}",
                )
            )
            continue

        loaded[name] = weight

    target_state.update(loaded)
    target_model.load_state_dict(target_state)

    return len(loaded), skipped


# ============================================================
# BATCH UTILS
# ============================================================

def unpack_raw_batch(batch):
    """
    Attend :
        loc, edge_attr, node_feat, loc_end, start_idx
    """

    if len(batch) != 5:
        raise ValueError(
            "Le batch doit contenir 5 éléments : "
            "loc, edge_attr, node_feat, loc_end, start_idx. "
            "Dans MD17Dataset.__getitem__, renvoie par exemple :\n"
            "return self.x_0[i], self.edge_attr, self.node_feat, self.x_t[i], self.st[i]"
        )

    loc, edge_attr, node_feat, loc_end, start_idx = batch

    return (
        loc.to(DEVICE),
        edge_attr.to(DEVICE),
        node_feat.to(DEVICE),
        loc_end.to(DEVICE),
        start_idx.to(DEVICE).long(),
    )


def prepare_tensors_from_raw(
    loc,
    edge_attr,
    node_feat,
    loc_end,
    start_idx,
    dataset,
):
    """
    loc       : (B, T, N, 3)
    edge_attr : (B, E, Fe)
    node_feat : (B, N, F)
    loc_end   : (B, N, 3)

    Retourne les tenseurs au format attendu par ESTAG.
    """

    batch_size = loc.shape[0]
    n_nodes = dataset.n_node

    loc = (
        loc.permute(1, 0, 2, 3)
           .reshape(NUM_PAST, batch_size * n_nodes, 3)
    )

    node_feat = node_feat.reshape(
        batch_size * n_nodes,
        -1,
    )

    edge_attr = edge_attr.reshape(
        -1,
        edge_attr.shape[-1],
    )

    loc_end = loc_end.reshape(
        batch_size * n_nodes,
        3,
    )

    edges = dataset.get_edges(
        batch_size,
        n_nodes,
    )

    edges = [e.to(DEVICE) for e in edges]

    return loc, edge_attr, node_feat, loc_end, start_idx, edges, batch_size


def valid_start_mask(dataset, start_idx, max_future_step):
    """
    step=1 correspond à :
        start + DELTA_FRAME * NUM_PAST

    step=s correspond à :
        start + DELTA_FRAME * (NUM_PAST + s - 1)
    """

    last_needed = start_idx + DELTA_FRAME * (
        NUM_PAST + max_future_step - 1
    )

    return last_needed < len(dataset.x)


def get_ground_truth_frames(dataset, target_indices):
    """
    target_indices : tensor shape (B,)

    Retourne :
        target : torch.Tensor, shape (B, N, 3)
    """

    if torch.is_tensor(dataset.x):

        idx = target_indices.detach().cpu().long()

        target = dataset.x[idx]

        return target.to(
            DEVICE,
            dtype=torch.float32,
        )

    else:

        idx = target_indices.detach().cpu().numpy().astype(int)

        target = dataset.x[idx]

        return torch.as_tensor(
            target,
            dtype=torch.float32,
            device=DEVICE,
        )


# ============================================================
# XYZ UTILS
# ============================================================

def get_symbols_from_dataset(dataset):
    """
    Essaie de récupérer les symboles atomiques du dataset.
    """

    if hasattr(dataset, "z"):

        z = dataset.z

        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        else:
            z = np.asarray(z)

    elif hasattr(dataset, "mole_idx"):

        z = dataset.mole_idx

        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        else:
            z = np.asarray(z)

    elif hasattr(dataset, "node_feat"):

        node_feat = dataset.node_feat

        if torch.is_tensor(node_feat):
            z = node_feat[:, 0].detach().cpu().numpy()
        else:
            z = np.asarray(node_feat)[:, 0]

    else:

        return ["X"] * dataset.n_node

    return [
        PTABLE.get(int(round(float(zi))), "X")
        for zi in z
    ]


def write_xyz(filename, symbols, frames, comments=None):
    """
    frames : list[np.ndarray], chaque frame est (N,3)
    """

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    with open(filename, "w") as f:

        for i, frame in enumerate(frames):

            f.write(f"{len(symbols)}\n")

            if comments is None:
                f.write(f"Frame {i}\n")
            else:
                f.write(f"{comments[i]}\n")

            for atom, xyz in zip(symbols, frame):

                x, y, z = xyz

                f.write(
                    f"{atom:2s} "
                    f"{x:.8f} "
                    f"{y:.8f} "
                    f"{z:.8f}\n"
                )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    dataset,
    loader,
    optimizer,
    criterion,
    rollout_steps,
):
    model.train()

    n_nodes = dataset.n_node

    weights = torch.full(
        (rollout_steps,),
        1.0 / rollout_steps,
        device=DEVICE,
    )

    epoch_loss = 0.0
    n_batches = 0

    for batch in loader:

        loc, edge_attr, node_feat, loc_end, start_idx = unpack_raw_batch(batch)

        mask = valid_start_mask(
            dataset,
            start_idx,
            rollout_steps,
        )

        if not mask.any():
            continue

        loc = loc[mask]
        edge_attr = edge_attr[mask]
        node_feat = node_feat[mask]
        loc_end = loc_end[mask]
        start_idx = start_idx[mask]

        loc, edge_attr, node_feat, loc_end, start_idx, edges, batch_size = prepare_tensors_from_raw(
            loc,
            edge_attr,
            node_feat,
            loc_end,
            start_idx,
            dataset,
        )

        optimizer.zero_grad()

        current_window = loc.clone()

        rollout_loss = 0.0

        for k in range(rollout_steps):

            pred = model(
                node_feat,
                current_window,
                edges,
                edge_attr,
            )

            target_indices = start_idx + DELTA_FRAME * (
                NUM_PAST + k
            )

            target = get_ground_truth_frames(
                dataset,
                target_indices,
            ).reshape(batch_size * n_nodes, 3)

            rollout_loss = rollout_loss + weights[k] * criterion(
                pred,
                target,
            )

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

    if n_batches == 0:
        return float("nan")

    return epoch_loss / n_batches


# ============================================================
# EVAL ONE-STEP
# ============================================================

@torch.no_grad()
def evaluate_one_step(
    model,
    dataset,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:

        loc, edge_attr, node_feat, loc_end, start_idx = unpack_raw_batch(batch)

        loc, edge_attr, node_feat, loc_end, start_idx, edges, batch_size = prepare_tensors_from_raw(
            loc,
            edge_attr,
            node_feat,
            loc_end,
            start_idx,
            dataset,
        )

        pred = model(
            node_feat,
            loc,
            edges,
            edge_attr,
        )

        loss = criterion(
            pred,
            loc_end,
        )

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return float("nan")

    return total_loss / n_batches


# ============================================================
# EVAL ROLLOUT LONG
# ============================================================

@torch.no_grad()
def evaluate_rollout(
    model,
    dataset,
    loader,
    rollout_steps_to_measure,
    result_dir,
    stage_tag,
    molecule,
    divergence_threshold=0.1,
    divergence_max_step=2000,
    save_xyz=True,
    max_saved_trajs=5,
):
    """
    Évalue le rollout.

    Mesure :
        - MSE aux steps indiqués.
        - première frame où MSE > divergence_threshold.

    Sauvegarde quelques trajectoires prédite/référence en XYZ.
    """

    model.eval()

    n_nodes = dataset.n_node
    symbols = get_symbols_from_dataset(dataset)

    max_eval_step = max(rollout_steps_to_measure)
    max_step = max(max_eval_step, divergence_max_step)

    sums = {
        step: 0.0
        for step in rollout_steps_to_measure
    }

    counts = {
        step: 0
        for step in rollout_steps_to_measure
    }

    divergence_rows = []

    xyz_dir = os.path.join(
        result_dir,
        "xyz_rollouts",
        stage_tag,
    )

    if save_xyz:
        os.makedirs(xyz_dir, exist_ok=True)

    sample_counter = 0
    saved_count = 0

    for batch in loader:

        loc, edge_attr, node_feat, loc_end, start_idx = unpack_raw_batch(batch)

        if loc.shape[0] != 1:
            raise ValueError(
                "evaluate_rollout doit recevoir un loader avec batch_size=1."
            )

        if not valid_start_mask(dataset, start_idx, max_step).all():
            continue

        start_idx_int = int(start_idx.item())

        loc, edge_attr, node_feat, loc_end, start_idx, edges, batch_size = prepare_tensors_from_raw(
            loc,
            edge_attr,
            node_feat,
            loc_end,
            start_idx,
            dataset,
        )

        current_window = loc.clone()

        first_divergence_step = -1
        first_divergence_abs_frame = -1

        pred_xyz_frames = []
        true_xyz_frames = []
        comments = []

        initial_abs_frame = start_idx_int + DELTA_FRAME * (
            NUM_PAST - 1
        )

        initial_frame = (
            current_window[-1]
            .reshape(n_nodes, 3)
            .detach()
            .cpu()
            .numpy()
        )

        should_save_this = (
            save_xyz
            and saved_count < max_saved_trajs
        )

        if should_save_this:

            pred_xyz_frames.append(initial_frame.copy())
            true_xyz_frames.append(initial_frame.copy())

            comments.append(
                f"{stage_tag} {molecule} sample={sample_counter} "
                f"step=0 abs_frame={initial_abs_frame}"
            )

        for step in range(1, max_step + 1):

            pred = model(
                node_feat,
                current_window,
                edges,
                edge_attr,
            )

            target_indices = start_idx + DELTA_FRAME * (
                NUM_PAST + step - 1
            )

            target = get_ground_truth_frames(
                dataset,
                target_indices,
            ).reshape(batch_size * n_nodes, 3)

            mse = torch.mean(
                (pred - target) ** 2
            ).item()

            if step in rollout_steps_to_measure:

                sums[step] += mse
                counts[step] += 1

            if (
                first_divergence_step == -1
                and mse > divergence_threshold
            ):

                first_divergence_step = step
                first_divergence_abs_frame = int(target_indices.item())

            if should_save_this:

                pred_frame = (
                    pred.reshape(n_nodes, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )

                true_frame = (
                    target.reshape(n_nodes, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )

                pred_xyz_frames.append(pred_frame.copy())
                true_xyz_frames.append(true_frame.copy())

                comments.append(
                    f"{stage_tag} {molecule} sample={sample_counter} "
                    f"step={step} "
                    f"abs_frame={int(target_indices.item())} "
                    f"mse={mse:.8f}"
                )

            current_window = torch.cat(
                [
                    current_window[1:],
                    pred.unsqueeze(0),
                ],
                dim=0,
            )

            # On continue au moins jusqu'au plus grand step d'évaluation.
            # Ensuite, dès que la divergence a été trouvée, on peut arrêter.
            if (
                step >= max_eval_step
                and first_divergence_step != -1
            ):
                break

        divergence_rows.append(
            {
                "sample": sample_counter,
                "start_idx": start_idx_int,
                "first_divergence_step": first_divergence_step,
                "first_divergence_abs_frame": first_divergence_abs_frame,
            }
        )

        if should_save_this:

            base = (
                f"{stage_tag}_"
                f"sample_{sample_counter:04d}_"
                f"start_{start_idx_int}"
            )

            pred_file = os.path.join(
                xyz_dir,
                base + "_pred.xyz",
            )

            true_file = os.path.join(
                xyz_dir,
                base + "_true.xyz",
            )

            write_xyz(
                pred_file,
                symbols,
                pred_xyz_frames,
                comments=comments,
            )

            write_xyz(
                true_file,
                symbols,
                true_xyz_frames,
                comments=comments,
            )

            saved_count += 1

        sample_counter += 1

    rollout_results = {}

    for step in rollout_steps_to_measure:

        if counts[step] == 0:
            rollout_results[step] = float("nan")
        else:
            rollout_results[step] = sums[step] / counts[step]

    divergence_csv = os.path.join(
        result_dir,
        f"{stage_tag}_divergence_frames.csv",
    )

    with open(divergence_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sample",
                "start_idx",
                "first_divergence_step",
                "first_divergence_abs_frame",
                "threshold_mse",
            ]
        )

        for row in divergence_rows:

            writer.writerow(
                [
                    row["sample"],
                    row["start_idx"],
                    row["first_divergence_step"],
                    row["first_divergence_abs_frame"],
                    divergence_threshold,
                ]
            )

    valid_divergences = [
        row["first_divergence_step"]
        for row in divergence_rows
        if row["first_divergence_step"] != -1
    ]

    n_not_diverged = sum(
        row["first_divergence_step"] == -1
        for row in divergence_rows
    )

    if len(valid_divergences) > 0:

        divergence_summary = {
            "mean_first_divergence_step": float(np.mean(valid_divergences)),
            "median_first_divergence_step": float(np.median(valid_divergences)),
            "min_first_divergence_step": int(np.min(valid_divergences)),
            "max_first_divergence_step": int(np.max(valid_divergences)),
            "n_not_diverged": int(n_not_diverged),
        }

    else:

        divergence_summary = {
            "mean_first_divergence_step": float("nan"),
            "median_first_divergence_step": float("nan"),
            "min_first_divergence_step": -1,
            "max_first_divergence_step": -1,
            "n_not_diverged": int(n_not_diverged),
        }

    return rollout_results, divergence_summary


# ============================================================
# LOGGING
# ============================================================

def init_log_file():

    if os.path.isfile(LOG_FILE):
        return

    with open(LOG_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "stage",
                "stage_tag",
                "molecule",
                "epoch",
                "train_loss",
                "test_mse_one_step",
                "rollout_mse_step_11",
                "rollout_mse_step_100",
                "rollout_mse_step_200",
                "rollout_mse_step_1000",
                "mean_first_divergence_step",
                "median_first_divergence_step",
                "min_first_divergence_step",
                "max_first_divergence_step",
                "n_not_diverged",
                "checkpoint",
            ]
        )


def append_log(
    stage,
    stage_tag,
    molecule,
    epoch,
    train_loss,
    test_mse,
    rollout_metrics,
    divergence_summary,
    checkpoint_path,
):

    with open(LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                stage,
                stage_tag,
                molecule,
                epoch,
                train_loss,
                test_mse,
                rollout_metrics.get(11, float("nan")),
                rollout_metrics.get(100, float("nan")),
                rollout_metrics.get(200, float("nan")),
                rollout_metrics.get(1000, float("nan")),
                divergence_summary["mean_first_divergence_step"],
                divergence_summary["median_first_divergence_step"],
                divergence_summary["min_first_divergence_step"],
                divergence_summary["max_first_divergence_step"],
                divergence_summary["n_not_diverged"],
                checkpoint_path,
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    init_log_file()

    previous_state = None

    for stage_id, molecule in enumerate(MOLECULE_SEQUENCE, start=1):

        stage_tag = f"stage_{stage_id:02d}_{molecule}"

        print()
        print("=" * 80)
        print(f"{stage_tag}")
        print("=" * 80)

        train_dataset = make_dataset(
            molecule=molecule,
            partition="train",
            max_samples=TRAIN_SAMPLES,
        )

        test_dataset = make_dataset(
            molecule=molecule,
            partition="test",
            max_samples=TEST_SAMPLES,
        )

        n_nodes = train_dataset.n_node

        print("n_nodes :", n_nodes)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        rollout_test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        model = make_model(
            n_nodes=n_nodes,
        )

        print("Nombre de paramètres :", count_parameters(model))

        if previous_state is not None:

            n_loaded, skipped = transfer_compatible_weights(
                previous_state,
                model,
            )

            print("Poids transférés :", n_loaded)
            print("Poids ignorés    :", len(skipped))

            skipped_file = os.path.join(
                RESULT_DIR,
                f"{stage_tag}_skipped_weights.txt",
            )

            with open(skipped_file, "w") as f:

                for name, reason in skipped:
                    f.write(f"{name} | {reason}\n")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        criterion = nn.MSELoss()

        best_test_mse = float("inf")
        best_state = None
        best_checkpoint_path = None

        for epoch in range(1, EPOCHS_PER_MOL + 1):

            if epoch <= 30:
                rollout_steps = 1
            elif epoch <= 40:
                rollout_steps = 2
            elif epoch <= 50:
                rollout_steps = 4
            elif epoch <= 60:
                rollout_steps = 6
            elif epoch <= 70:
                rollout_steps = 8
            elif epoch <= 80:
                rollout_steps = 10
            else:
                rollout_steps = NUM_PAST + 1  # 11

            train_loss = train_one_epoch(
                model=model,
                dataset=train_dataset,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                rollout_steps=rollout_steps,
            )

            do_eval = (
                epoch == 1
                or epoch % EVAL_EVERY == 0
                or epoch == EPOCHS_PER_MOL
            )

            if not do_eval:

                print(
                    f"{stage_tag:20s} | "
                    f"epoch {epoch:4d} | "
                    f"train = {train_loss:.8f} | "
                    f"rollout_train = {rollout_steps:2d} | "
                )

                continue

            test_mse = evaluate_one_step(
                model=model,
                dataset=test_dataset,
                loader=test_loader,
                criterion=criterion,
            )

            do_rollout_eval = (
                not ROLLOUT_EVAL_ONLY_LAST_EPOCH
                or epoch == EPOCHS_PER_MOL
            )

            if do_rollout_eval:

                rollout_metrics, divergence_summary = evaluate_rollout(
                    model=model,
                    dataset=test_dataset,
                    loader=rollout_test_loader,
                    rollout_steps_to_measure=EVAL_ROLLOUT_STEPS,
                    result_dir=RESULT_DIR,
                    stage_tag=stage_tag,
                    molecule=molecule,
                    divergence_threshold=DIVERGENCE_MSE_THRESHOLD,
                    divergence_max_step=DIVERGENCE_MAX_STEP,
                    save_xyz=SAVE_ROLLOUT_XYZ,
                    max_saved_trajs=MAX_SAVED_ROLLOUT_TRAJS,
                )

            else:

                rollout_metrics = {
                    11: float("nan"),
                    100: float("nan"),
                    200: float("nan"),
                    1000: float("nan"),
                }

                divergence_summary = {
                    "mean_first_divergence_step": float("nan"),
                    "median_first_divergence_step": float("nan"),
                    "min_first_divergence_step": -1,
                    "max_first_divergence_step": -1,
                    "n_not_diverged": -1,
                }

            checkpoint_path = os.path.join(
                RESULT_DIR,
                f"{stage_tag}_epoch_{epoch:04d}.pt",
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            append_log(
                stage=stage_id,
                stage_tag=stage_tag,
                molecule=molecule,
                epoch=epoch,
                train_loss=train_loss,
                test_mse=test_mse,
                rollout_metrics=rollout_metrics,
                divergence_summary=divergence_summary,
                checkpoint_path=checkpoint_path,
            )

            print(
                f"{stage_tag:20s} | "
                f"epoch {epoch:4d} | "
                f"train = {train_loss:.8f} | "
                f"test = {test_mse:.8f} | "
                f"r11 = {rollout_metrics[11]:.8f} | "
                f"r100 = {rollout_metrics[100]:.8f} | "
                f"r200 = {rollout_metrics[200]:.8f} | "
                f"r1000 = {rollout_metrics[1000]:.8f} | "
                f"div_mean = {divergence_summary['mean_first_divergence_step']}"
            )

            if test_mse < best_test_mse:

                best_test_mse = test_mse
                best_state = copy.deepcopy(model.state_dict())

                best_checkpoint_path = os.path.join(
                    RESULT_DIR,
                    f"{stage_tag}_best.pt",
                )

                torch.save(
                    best_state,
                    best_checkpoint_path,
                )

        last_checkpoint_path = os.path.join(
            RESULT_DIR,
            f"{stage_tag}_last.pt",
        )

        torch.save(
            model.state_dict(),
            last_checkpoint_path,
        )

        print()
        print("Best checkpoint :", best_checkpoint_path)
        print("Last checkpoint :", last_checkpoint_path)

        # On transfère le meilleur modèle de ce stage vers la molécule suivante.
        if best_state is not None:
            previous_state = best_state
        else:
            previous_state = copy.deepcopy(model.state_dict())

    print()
    print("Terminé.")
    print("Logs :", LOG_FILE)
    print("Résultats :", RESULT_DIR)


if __name__ == "__main__":
    main()