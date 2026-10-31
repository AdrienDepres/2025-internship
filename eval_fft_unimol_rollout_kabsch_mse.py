# eval_fft_unimol_rollout_kabsch_mse.py
#
# Évalue un modèle ESTAG + UniMol avec fft=True sur aspirin.
#
# Le script :
#   1) charge un checkpoint aspirin,
#   2) génère 100 rollouts depuis le split test MD17 aspirin,
#   3) ne corrige PAS les prédictions pendant le rollout,
#   4) avant le calcul de MSE, aligne par Kabsch :
#          pred_step   -> reference_frame
#          true_step   -> reference_frame
#      avec la même référence pour les deux,
#   5) calcule les MSE moyennes aux steps 1, 11, 100, 200,
#      en ignorant les NaN,
#   6) sauvegarde un CSV résumé et un CSV détaillé par sample.
#
# Hypothèse :
#   MD17Dataset.__getitem__ renvoie :
#       loc, edge_attr, node_feat, loc_end, start_idx


import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# CONFIG À MODIFIER
# ============================================================

CHECKPOINT = "resultats/estag_unimol_rollout_fft_aspirine.pt"
OUT_DIR = "resultats/centre/eval_fft_kabsch_aspirin"

DATA_DIR = "md17"
MOLECULE = "aspirin"

TARGET_N_ROLLOUTS = 100
TEST_MAX_SAMPLES = 2000

NUM_PAST = 10
DELTA_FRAME = 10

BATCH_SIZE = 8

MEASURE_STEPS = [1, 11, 100, 200]
MAX_ROLLOUT_STEP = max(MEASURE_STEPS)

# ESTAG + UniMol
IN_NODE_NF = 513
HIDDEN_NF = 32
N_LAYERS = 2

# FFT activé
USE_FFT = True
IN_EDGE_NF = NUM_PAST - 1
NODES_ATT_DIM = NUM_PAST - 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# KABSCH
# ============================================================

def kabsch_align_torch(mobile, reference):
    """
    Aligne mobile sur reference.

    mobile    : (B, N, 3)
    reference : (B, N, 3)

    Retourne :
        mobile_aligned : (B, N, 3)
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

    sign = torch.where(
        det < 0,
        torch.tensor(-1.0, device=mobile.device, dtype=mobile.dtype),
        torch.tensor(1.0, device=mobile.device, dtype=mobile.dtype),
    )

    D = torch.eye(
        3,
        device=mobile.device,
        dtype=mobile.dtype,
    ).unsqueeze(0).repeat(mobile.shape[0], 1, 1)

    D[:, 2, 2] = sign

    R = torch.matmul(
        torch.matmul(U, D),
        Vh,
    )

    mobile_aligned = torch.matmul(X, R) + reference_center

    return mobile_aligned


def align_flat_to_reference(frame, reference, batch_size, n_nodes):
    """
    frame     : (B*N, 3)
    reference : (B*N, 3)

    Retourne frame alignée sur reference, shape (B*N, 3).
    """

    frame_b = frame.view(batch_size, n_nodes, 3)
    ref_b = reference.view(batch_size, n_nodes, 3)

    aligned = kabsch_align_torch(
        frame_b,
        ref_b,
    )

    return aligned.reshape(batch_size * n_nodes, 3)


# ============================================================
# DATA UTILS
# ============================================================

def get_ground_truth_frames(dataset, target_indices):
    """
    target_indices : torch.Tensor shape (B,)
    Retourne : torch.Tensor shape (B, N, 3)
    """

    idx = target_indices.detach().cpu().numpy().astype(int)
    x = dataset.x[idx]

    return torch.as_tensor(
        x,
        dtype=torch.float32,
        device=DEVICE,
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


def prepare_batch(batch, dataset):
    """
    Attend :
        loc, edge_attr, node_feat, loc_end, start_idx
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

    edge_attr_batch = edge_attr.reshape(
        -1,
        edge_attr.shape[-1],
    )

    node_feat = node_feat.reshape(
        batch_size * n_nodes,
        -1,
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


# ============================================================
# MODEL
# ============================================================

def build_model(n_nodes):
    return ESTAG(
        num_past=NUM_PAST,
        num_future=1,
        in_node_nf=IN_NODE_NF,
        in_edge_nf=IN_EDGE_NF,
        hidden_nf=HIDDEN_NF,
        fft=USE_FFT,
        eat=True,
        device=DEVICE,
        n_layers=N_LAYERS,
        n_nodes=n_nodes,
        nodes_att_dim=NODES_ATT_DIM,
    ).to(DEVICE)


def load_checkpoint(model, checkpoint_path):
    """
    Compatible avec :
      - torch.save(model.state_dict())
      - torch.save({"state_dict": ...})
      - torch.save({"model_state_dict": ...})
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)

    return model


# ============================================================
# EVAL
# ============================================================

@torch.no_grad()
def evaluate_100_rollouts_with_kabsch_mse(model, dataset, loader):
    model.eval()

    n_nodes = dataset.n_node

    mse_values = {
        step: []
        for step in MEASURE_STEPS
    }

    per_sample_rows = []

    n_nan = {
        step: 0
        for step in MEASURE_STEPS
    }

    n_rollouts_done = 0
    n_skipped = 0

    for batch in loader:

        # Le batch arrive CPU depuis le DataLoader.
        # On filtre donc encore sur CPU.
        raw_start_idx = batch[-1].long()

        mask = valid_start_mask(
            dataset,
            raw_start_idx,
            MAX_ROLLOUT_STEP,
        )

        if not mask.any():
            n_skipped += raw_start_idx.numel()
            continue

        batch = [
            x[mask]
            for x in batch
        ]

        remaining = TARGET_N_ROLLOUTS - n_rollouts_done

        if remaining <= 0:
            break

        if batch[0].shape[0] > remaining:
            batch = [
                x[:remaining]
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

        n_rollouts_done += batch_size

        # Référence commune pour tout le rollout :
        # dernière frame d'entrée.
        # On aligne pred et true sur CETTE référence avant MSE.
        reference_frame = current_window[-1].clone()

        sample_mse_record = {
            step: torch.full(
                (batch_size,),
                float("nan"),
                device=DEVICE,
            )
            for step in MEASURE_STEPS
        }

        for step in range(1, MAX_ROLLOUT_STEP + 1):

            pred_raw = model(
                node_feat,
                current_window,
                edges,
                edge_attr_batch,
            )

            target_indices = start_idx + DELTA_FRAME * (
                NUM_PAST + step - 1
            )

            target_raw = get_ground_truth_frames(
                dataset,
                target_indices,
            ).reshape(batch_size * n_nodes, 3)

            if step in MEASURE_STEPS:

                pred_aligned = align_flat_to_reference(
                    pred_raw,
                    reference_frame,
                    batch_size,
                    n_nodes,
                )

                target_aligned = align_flat_to_reference(
                    target_raw,
                    reference_frame,
                    batch_size,
                    n_nodes,
                )

                per_sample_mse = (
                    (pred_aligned - target_aligned)
                    .reshape(batch_size, n_nodes, 3)
                    .pow(2)
                    .mean(dim=(1, 2))
                )

                sample_mse_record[step] = per_sample_mse.detach()

                finite = torch.isfinite(per_sample_mse)

                if finite.any():
                    mse_values[step].extend(
                        per_sample_mse[finite]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                n_nan[step] += int(
                    (~finite).sum().item()
                )

            # Rollout BRUT, non corrigé.
            # Kabsch sert uniquement pour la métrique.
            current_window = torch.cat(
                [
                    current_window[1:],
                    pred_raw.unsqueeze(0),
                ],
                dim=0,
            )

        for i in range(batch_size):

            row = {
                "sample_id": n_rollouts_done - batch_size + i,
                "start_idx": int(start_idx[i].item()),
            }

            for step in MEASURE_STEPS:
                row[f"mse_step_{step}"] = float(
                    sample_mse_record[step][i].detach().cpu()
                )

            per_sample_rows.append(row)

        if n_rollouts_done >= TARGET_N_ROLLOUTS:
            break

    summary = {}

    for step in MEASURE_STEPS:

        vals = np.asarray(
            mse_values[step],
            dtype=np.float64,
        )

        if len(vals) == 0:
            mean_mse = float("nan")
            median_mse = float("nan")
        else:
            mean_mse = float(np.mean(vals))
            median_mse = float(np.median(vals))

        summary[step] = {
            "mean_mse": mean_mse,
            "median_mse": median_mse,
            "n_finite": int(len(vals)),
            "n_nan": int(n_nan[step]),
        }

    return summary, per_sample_rows, n_rollouts_done, n_skipped


# ============================================================
# CSV
# ============================================================

def save_summary_csv(summary, n_rollouts_done, n_skipped):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(
        OUT_DIR,
        "summary_kabsch_mse.csv",
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "step",
                "mean_mse",
                "median_mse",
                "n_finite",
                "n_nan",
                "n_rollouts_done",
                "n_skipped",
                "checkpoint",
                "fft",
                "kabsch_reference",
            ]
        )

        for step in MEASURE_STEPS:
            writer.writerow(
                [
                    step,
                    summary[step]["mean_mse"],
                    summary[step]["median_mse"],
                    summary[step]["n_finite"],
                    summary[step]["n_nan"],
                    n_rollouts_done,
                    n_skipped,
                    CHECKPOINT,
                    USE_FFT,
                    "last_input_frame",
                ]
            )

    return path


def save_per_sample_csv(per_sample_rows):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(
        OUT_DIR,
        "per_sample_kabsch_mse.csv",
    )

    fieldnames = [
        "sample_id",
        "start_idx",
    ] + [
        f"mse_step_{step}"
        for step in MEASURE_STEPS
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in per_sample_rows:
            writer.writerow(row)

    return path


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = MD17Dataset(
        partition="test",
        max_samples=TEST_MAX_SAMPLES,
        data_dir=DATA_DIR,
        molecule_type=MOLECULE,
        delta_frame=DELTA_FRAME,
        num_past=NUM_PAST,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        n_nodes=dataset.n_node,
    )

    model = load_checkpoint(
        model,
        CHECKPOINT,
    )

    model.eval()

    print("Device       :", DEVICE)
    print("Checkpoint   :", CHECKPOINT)
    print("Out dir      :", OUT_DIR)
    print("Molecule     :", MOLECULE)
    print("n_nodes      :", dataset.n_node)
    print("FFT          :", USE_FFT)
    print("in_edge_nf   :", IN_EDGE_NF)
    print("nodes_att_dim:", NODES_ATT_DIM)
    print("Target N     :", TARGET_N_ROLLOUTS)
    print("Steps        :", MEASURE_STEPS)
    print()

    summary, per_sample_rows, n_rollouts_done, n_skipped = evaluate_100_rollouts_with_kabsch_mse(
        model=model,
        dataset=dataset,
        loader=loader,
    )

    summary_csv = save_summary_csv(
        summary,
        n_rollouts_done,
        n_skipped,
    )

    per_sample_csv = save_per_sample_csv(
        per_sample_rows,
    )

    print("Résultats MSE après Kabsch commun vers la dernière frame d'entrée")
    print("Moyennes hors NaN")
    print()

    for step in MEASURE_STEPS:
        print(
            f"step {step:4d} | "
            f"mean = {summary[step]['mean_mse']:.8e} | "
            f"median = {summary[step]['median_mse']:.8e} | "
            f"n_finite = {summary[step]['n_finite']} | "
            f"n_nan = {summary[step]['n_nan']}"
        )

    print()
    print("rollouts faits :", n_rollouts_done)
    print("samples ignorés:", n_skipped)
    print("CSV résumé     :", summary_csv)
    print("CSV détaillé   :", per_sample_csv)


if __name__ == "__main__":
    main()
