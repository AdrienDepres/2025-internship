# eval_rollout_aspirin_aligned.py

import os
import csv
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# CONFIG PAR DÉFAUT
# ============================================================

DEFAULT_CHECKPOINT = "resultats/centre/estag_unimol_rollout_prog_aligne_targetv2.pt"
DEFAULT_OUT_DIR = "resultats/centre/"

NUM_PAST = 10
DELTA_FRAME = 10

BATCH_SIZE = 32

DATA_DIR = "md17"
MOLECULE = "aspirin"

MAX_SAMPLES = 1000

MEASURE_STEPS = [1, 11, 100, 200]
MAX_ROLLOUT_STEP = max(MEASURE_STEPS)

IN_NODE_NF = 513
IN_EDGE_NF = 3
HIDDEN_NF = 32
N_LAYERS = 2

# "kabsch" = translation + rotation globale
# "translation" = recentrage du centre géométrique seulement
# "none" = aucun recentrage
ALIGN_MODE = "kabsch"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# ALIGNEMENT
# ============================================================

def kabsch_align_torch(mobile, reference):
    """
    mobile, reference : (B, N, 3)

    Aligne mobile sur reference par translation + rotation optimale.
    """

    mobile_center = mobile.mean(dim=1, keepdim=True)
    reference_center = reference.mean(dim=1, keepdim=True)

    X = mobile - mobile_center
    Y = reference - reference_center

    C = torch.matmul(
        X.transpose(1, 2),
        Y,
    )

    U, S, Vh = torch.linalg.svd(C)

    det = torch.det(
        torch.matmul(U, Vh)
    )

    D = torch.eye(
        3,
        device=mobile.device,
        dtype=mobile.dtype,
    ).unsqueeze(0).repeat(mobile.shape[0], 1, 1)

    D[:, 2, 2] = torch.sign(det)

    R = torch.matmul(
        torch.matmul(U, D),
        Vh,
    )

    aligned = torch.matmul(X, R) + reference_center

    return aligned


def center_to_reference(mobile, reference):
    """
    mobile, reference : (B, N, 3)

    Recentrage par translation seule.
    """

    mobile_center = mobile.mean(dim=1, keepdim=True)
    reference_center = reference.mean(dim=1, keepdim=True)

    return mobile - mobile_center + reference_center


def align_frame(frame, reference, batch_size, n_nodes, mode):
    """
    frame     : (B*N, 3)
    reference : (B*N, 3)

    Retourne frame alignée sur reference, shape (B*N, 3).
    """

    if mode == "none":
        return frame

    frame_b = frame.view(batch_size, n_nodes, 3)
    ref_b = reference.view(batch_size, n_nodes, 3)

    if mode == "kabsch":
        aligned = kabsch_align_torch(
            frame_b,
            ref_b,
        )

    elif mode == "translation":
        aligned = center_to_reference(
            frame_b,
            ref_b,
        )

    else:
        raise ValueError(f"ALIGN_MODE inconnu : {mode}")

    return aligned.reshape(batch_size * n_nodes, 3)


# ============================================================
# XYZ
# ============================================================

def get_symbols_from_dataset(dataset):
    if hasattr(dataset, "z"):
        z = dataset.z
    elif hasattr(dataset, "mole_idx"):
        z = dataset.mole_idx
    elif hasattr(dataset, "node_feat"):
        z = dataset.node_feat[:, 0]
    else:
        return ["X"] * dataset.n_node

    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()
    else:
        z = np.asarray(z)

    return [
        PTABLE.get(int(round(float(zi))), "X")
        for zi in z
    ]


def write_xyz(filename, symbols, frames, comments=None):
    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    with open(filename, "w") as f:

        for i, frame in enumerate(frames):

            f.write(f"{len(symbols)}\n")

            if comments is None:
                f.write(f"frame={i}\n")
            else:
                f.write(comments[i] + "\n")

            for atom, xyz in zip(symbols, frame):

                x, y, z = xyz

                f.write(
                    f"{atom:2s} "
                    f"{x:.8f} "
                    f"{y:.8f} "
                    f"{z:.8f}\n"
                )


# ============================================================
# DATA UTILS
# ============================================================

def get_ground_truth_frames(dataset, target_indices):
    """
    target_indices : torch.Tensor shape (B,)

    Retourne : (B, N, 3)
    """

    idx = target_indices.detach().cpu().numpy().astype(int)

    target_np = dataset.x[idx]

    return torch.as_tensor(
        target_np,
        dtype=torch.float32,
        device=DEVICE,
    )


def prepare_batch(batch, dataset):
    """
    Attend :
        loc, edge_attr, node_feat, loc_end, start_idx

    Retourne :
        current_window : (T, B*N, 3)
        edge_attr_batch : (B*E, 3)
        node_feat : (B*N, F)
        start_idx : (B,)
        edges
        batch_size
    """

    if len(batch) != 5:
        raise ValueError(
            "Le dataset doit renvoyer 5 éléments : "
            "loc, edge_attr, node_feat, loc_end, start_idx."
        )

    loc, edge_attr, node_feat, _, start_idx = [
        x.to(DEVICE)
        for x in batch
    ]

    start_idx = start_idx.long()

    batch_size = loc.shape[0]
    n_nodes = dataset.n_node

    current_window = (
        loc.permute(1, 0, 2, 3)
        .reshape(NUM_PAST, batch_size * n_nodes, 3)
    )

    node_feat = node_feat.reshape(
        batch_size * n_nodes,
        -1,
    )

    edge_attr_batch = edge_attr.reshape(
        -1,
        edge_attr.shape[-1],
    )

    edges = dataset.get_edges(
        batch_size,
        n_nodes,
    )

    edges = [e.to(DEVICE) for e in edges]

    return (
        current_window,
        edge_attr_batch,
        node_feat,
        start_idx,
        edges,
        batch_size,
    )


def valid_start_mask(dataset, start_idx, max_step):
    """
    step=1 correspond à :
        start_idx + DELTA_FRAME * NUM_PAST
    """

    last_needed_idx = start_idx + DELTA_FRAME * (
        NUM_PAST + max_step - 1
    )

    return last_needed_idx < len(dataset.x)


# ============================================================
# MODELE
# ============================================================

def build_model(n_nodes):
    model = ESTAG(
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

    return model


# ============================================================
# EVAL ROLLOUT
# ============================================================

@torch.no_grad()
def evaluate_rollout(
    model,
    dataset,
    loader,
    out_dir,
    align_mode="kabsch",
    save_first_rollout=True,
):
    model.eval()

    n_nodes = dataset.n_node
    symbols = get_symbols_from_dataset(dataset)

    mse_values = {
        step: []
        for step in MEASURE_STEPS
    }

    first_rollout_saved = False

    pred_xyz_frames = []
    true_xyz_frames = []
    xyz_comments = []

    n_valid_samples = 0
    n_skipped_samples = 0
    n_nan_by_step = {
        step: 0
        for step in MEASURE_STEPS
    }

    for batch in loader:

        raw_start_idx = batch[-1].long()

        mask = valid_start_mask(
            dataset,
            raw_start_idx,
            MAX_ROLLOUT_STEP,

        )
        mask = mask.cpu()

        # if not mask.any():
        #     n_skipped_samples += raw_start_idx.numel()
        #     continue

        # Filtrage du batch brut
        batch = [
            x[mask]
            for x in batch
        ]

        (
            current_window,
            edge_attr_batch,
            node_feat,
            start_idx,
            edges,
            batch_size,
        ) = prepare_batch(
            batch,
            dataset,
        )

        n_valid_samples += batch_size

        save_this_batch = (
            save_first_rollout
            and not first_rollout_saved
        )

        if save_this_batch:

            initial_frame = (
                current_window[-1]
                .reshape(batch_size, n_nodes, 3)[0]
                .detach()
                .cpu()
                .numpy()
            )

            initial_abs_frame = int(
                start_idx[0].item()
                + DELTA_FRAME * (NUM_PAST - 1)
            )

            pred_xyz_frames.append(initial_frame.copy())
            true_xyz_frames.append(initial_frame.copy())

            xyz_comments.append(
                f"step=0 abs_frame={initial_abs_frame} align={align_mode}"
            )

        for step in range(1, MAX_ROLLOUT_STEP + 1):

            pred_raw = model(
                node_feat,
                current_window,
                edges,
                edge_attr_batch,
            )

            reference_frame = current_window[-1]

            pred = align_frame(
                pred_raw,
                reference_frame,
                batch_size,
                n_nodes,
                align_mode,
            )

            target_indices = start_idx + DELTA_FRAME * (
                NUM_PAST + step - 1
            )

            target_raw = get_ground_truth_frames(
                dataset,
                target_indices,
            ).reshape(batch_size * n_nodes, 3)

            # Important :
            # la référence est la même pour pred et target.
            target = align_frame(
                target_raw,
                reference_frame,
                batch_size,
                n_nodes,
                align_mode,
            )

            if step in MEASURE_STEPS:

                per_sample_mse = (
                    (pred - target)
                    .reshape(batch_size, n_nodes, 3)
                    .pow(2)
                    .mean(dim=(1, 2))
                )

                finite_mask = torch.isfinite(per_sample_mse)

                if finite_mask.any():
                    mse_values[step].extend(
                        per_sample_mse[finite_mask]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                n_nan_by_step[step] += int(
                    (~finite_mask).sum().item()
                )

            if save_this_batch:

                pred_frame = (
                    pred.reshape(batch_size, n_nodes, 3)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                true_frame = (
                    target.reshape(batch_size, n_nodes, 3)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                mse_first = float(
                    (
                        (pred.reshape(batch_size, n_nodes, 3)[0]
                         - target.reshape(batch_size, n_nodes, 3)[0])
                        .pow(2)
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

                abs_frame = int(target_indices[0].item())

                pred_xyz_frames.append(pred_frame.copy())
                true_xyz_frames.append(true_frame.copy())

                xyz_comments.append(
                    f"step={step} abs_frame={abs_frame} "
                    f"mse={mse_first:.8e} align={align_mode}"
                )

            # Réinjection de la prédiction recentrée/alignée.
            current_window = torch.cat(
                [
                    current_window[1:],
                    pred.unsqueeze(0),
                ],
                dim=0,
            )

        if save_this_batch:

            pred_file = os.path.join(
                out_dir,
                "first_rollout_pred_aligned.xyz",
            )

            true_file = os.path.join(
                out_dir,
                "first_rollout_true_aligned.xyz",
            )

            write_xyz(
                pred_file,
                symbols,
                pred_xyz_frames,
                comments=xyz_comments,
            )

            write_xyz(
                true_file,
                symbols,
                true_xyz_frames,
                comments=xyz_comments,
            )

            first_rollout_saved = True

    summary = {}

    for step in MEASURE_STEPS:

        vals = np.asarray(
            mse_values[step],
            dtype=np.float64,
        )

        if len(vals) == 0:
            mean_mse = float("nan")
        else:
            mean_mse = float(np.mean(vals))

        summary[step] = {
            "mean_mse": mean_mse,
            "n_finite": int(len(vals)),
            "n_nan": int(n_nan_by_step[step]),
        }

    os.makedirs(out_dir, exist_ok=True)

    csv_file = os.path.join(
        out_dir,
        "rollout_mse_summary.csv",
    )

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "step",
                "mean_mse",
                "n_finite",
                "n_nan",
                "align_mode",
            ]
        )

        for step in MEASURE_STEPS:

            writer.writerow(
                [
                    step,
                    summary[step]["mean_mse"],
                    summary[step]["n_finite"],
                    summary[step]["n_nan"],
                    align_mode,
                ]
            )

    return summary, n_valid_samples, n_skipped_samples


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--align_mode",
        type=str,
        default="kabsch",
        choices=["kabsch", "translation", "none"],
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=MAX_SAMPLES,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    dataset = MD17Dataset(
        partition="test",
        max_samples=args.max_samples,
        data_dir=DATA_DIR,
        molecule_type=MOLECULE,
        delta_frame=DELTA_FRAME,
        num_past=NUM_PAST,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_nodes = dataset.n_node

    model = build_model(
        n_nodes=n_nodes,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    print("Device       :", DEVICE)
    print("Checkpoint   :", args.checkpoint)
    print("Output dir   :", args.out_dir)
    print("Align mode   :", args.align_mode)
    print("n_nodes      :", n_nodes)
    print("max_samples  :", args.max_samples)
    print("measure steps:", MEASURE_STEPS)

    summary, n_valid, n_skipped = evaluate_rollout(
        model=model,
        dataset=dataset,
        loader=loader,
        out_dir=args.out_dir,
        align_mode=args.align_mode,
        save_first_rollout=True,
    )

    print()
    print("Résultats rollout, moyennes hors NaN :")
    print(f"valid samples   : {n_valid}")
    print(f"skipped samples : {n_skipped}")
    print()

    for step in MEASURE_STEPS:

        print(
            f"step {step:4d} | "
            f"MSE moyenne = {summary[step]['mean_mse']:.8e} | "
            f"n_finite = {summary[step]['n_finite']} | "
            f"n_nan = {summary[step]['n_nan']}"
        )

    print()
    print(
        "XYZ sauvegardés :",
        os.path.join(args.out_dir, "first_rollout_pred_aligned.xyz"),
        "et",
        os.path.join(args.out_dir, "first_rollout_true_aligned.xyz"),
    )
    print(
        "CSV sauvegardé :",
        os.path.join(args.out_dir, "rollout_mse_summary.csv"),
    )


if __name__ == "__main__":
    main()
