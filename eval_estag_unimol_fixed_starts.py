#!/usr/bin/env python3
"""
Évalue un checkpoint ESTAG + Uni-Mol sur des départs de rollout imposés.

Le script :
  1. charge le split test de MD17 pour l'aspirine ;
  2. retrouve exactement les départs indiqués dans un fichier texte ;
  3. effectue un rollout brut, sans Kabsch ni recentrage ;
  4. calcule la MSE à chaque pas pour chaque rollout ;
  5. écrit :
       - une matrice CSV compatible avec les fichiers déjà utilisés ;
       - la même matrice au format NPY ;
       - un CSV résumé par pas ;
       - une nouvelle colonne dans mse_summary_by_step.csv.

Le fichier de départs peut contenir :
  - les indices absolus start_idx renvoyés par MD17Dataset ;
  - ou les indices des éléments du split test.

Par défaut, le modèle est construit avec les paramètres du rapport :
  T = 10, delta_frame = 10, in_node_nf = 513,
  hidden_nf = 32, n_layers = 2, fft = True, eat = True.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Évalue ESTAG + Uni-Mol sur des départs fixes et sauvegarde "
            "la MSE de chaque rollout à chaque pas."
        )
    )

    parser.add_argument("--checkpoint", type=Path, default = "./resultats/estag_unimol_ptet/saved_model_state_dict.pt")
    parser.add_argument(
        "--fixed-starts",
        type=Path,
        default=Path("/home/tufdeb/Documents/ENS/stage/modeles/resultats/rollout_courbes_aspirine_rapport/fixed_starts_aspirin_100_max600.txt"),
    )

    parser.add_argument("--data-dir", type=str, default="md17")
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Même valeur que lors des évaluations précédentes.",
    )

    parser.add_argument("--num-past", type=int, default=10)
    parser.add_argument("--delta-frame", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--in-node-nf", type=int, default=513)
    parser.add_argument("--hidden-nf", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)

    parser.add_argument(
        "--fft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Employer --no-fft pour un checkpoint entraîné avec fft=False.",
    )
    parser.add_argument(
        "--eat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mse_estag_unimol_sans_rollout",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("mse_summary_by_step.csv"),
    )
    parser.add_argument(
        "--summary-column",
        type=str,
        default="estag_unimol_sans_rollout",
    )
    parser.add_argument("--no-update-summary", action="store_true")
    parser.add_argument("--non-strict", action="store_true")

    return parser.parse_args()


def read_fixed_starts(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de départs introuvable : {path}")

    starts: list[int] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0]
        starts.extend(int(value) for value in re.findall(r"-?\d+", line))

    if not starts:
        raise ValueError(f"Aucun entier trouvé dans {path}")

    if len(set(starts)) != len(starts):
        raise ValueError("Le fichier de départs contient des doublons.")

    return starts


def dataset_start_indices(dataset: MD17Dataset) -> np.ndarray:
    if hasattr(dataset, "st"):
        starts = np.asarray(dataset.st).reshape(-1)
        if len(starts) == len(dataset):
            return starts.astype(np.int64)

    starts = np.empty(len(dataset), dtype=np.int64)
    print("Lecture des start_idx via MD17Dataset.__getitem__...")

    for index in range(len(dataset)):
        item = dataset[index]
        if len(item) != 5:
            raise ValueError(
                "MD17Dataset.__getitem__ doit renvoyer "
                "loc, edge_attr, node_feat, loc_end, start_idx."
            )
        start = item[-1]
        starts[index] = int(start.item() if torch.is_tensor(start) else start)

    return starts


def resolve_subset_indices(
    dataset: MD17Dataset,
    fixed_values: list[int],
    max_steps: int,
    num_past: int,
    delta_frame: int,
) -> tuple[list[int], list[int], str]:
    starts_in_dataset = dataset_start_indices(dataset)

    positions_by_start: dict[int, deque[int]] = defaultdict(deque)
    for dataset_index, start in enumerate(starts_in_dataset):
        positions_by_start[int(start)].append(dataset_index)

    if all(value in positions_by_start for value in fixed_values):
        subset_indices = [
            positions_by_start[value].popleft() for value in fixed_values
        ]
        absolute_starts = fixed_values.copy()
        interpretation = "start_idx absolus"
    elif all(0 <= value < len(dataset) for value in fixed_values):
        subset_indices = fixed_values.copy()
        absolute_starts = [
            int(starts_in_dataset[index]) for index in subset_indices
        ]
        interpretation = "indices d'éléments du split test"
    else:
        missing = [
            value for value in fixed_values if value not in positions_by_start
        ]
        raise ValueError(
            "Départs introuvables dans le split test.\n"
            f"Premières valeurs non trouvées : {missing[:10]}\n"
            "Vérifie --max-samples, --partition, --num-past et --delta-frame."
        )

    trajectory_length = len(dataset.x)
    invalid = []

    for dataset_index, start in zip(subset_indices, absolute_starts):
        last_needed = start + delta_frame * (num_past + max_steps - 1)
        if last_needed >= trajectory_length:
            invalid.append((dataset_index, start, last_needed))

    if invalid:
        lines = "\n".join(
            f"  dataset_index={idx}, start_idx={start}, frame requise={last}"
            for idx, start, last in invalid[:10]
        )
        raise ValueError(
            f"{len(invalid)} départ(s) ne permettent pas {max_steps} pas :\n"
            f"{lines}\nLongueur de la trajectoire : {trajectory_length}"
        )

    return subset_indices, absolute_starts, interpretation


def prepare_batch(batch, dataset: MD17Dataset, num_past: int):
    if len(batch) != 5:
        raise ValueError(
            "Le batch doit contenir loc, edge_attr, node_feat, loc_end, start_idx."
        )

    loc, edge_attr, node_feat, _, start_idx = batch

    loc = loc.to(DEVICE)
    edge_attr = edge_attr.to(DEVICE)
    node_feat = node_feat.to(DEVICE)
    start_idx = start_idx.to(DEVICE).long()

    batch_size = loc.shape[0]
    n_nodes = dataset.n_node

    current_window = (
        loc.permute(1, 0, 2, 3)
        .reshape(num_past, batch_size * n_nodes, 3)
    )
    edge_attr_batch = edge_attr.reshape(-1, edge_attr.shape[-1])
    node_feat_batch = node_feat.reshape(batch_size * n_nodes, -1)

    edges = dataset.get_edges(batch_size, n_nodes)
    edges = [edge.to(DEVICE) for edge in edges]

    return (
        current_window,
        edge_attr_batch,
        node_feat_batch,
        start_idx,
        edges,
        batch_size,
    )


def get_ground_truth(dataset: MD17Dataset, indices: torch.Tensor) -> torch.Tensor:
    indices_np = indices.detach().cpu().numpy().astype(np.int64)
    coordinates = dataset.x[indices_np]
    return torch.as_tensor(coordinates, dtype=torch.float32, device=DEVICE)


def build_model(dataset: MD17Dataset, args: argparse.Namespace) -> ESTAG:
    if args.fft:
        in_edge_nf = args.num_past - 1
        nodes_att_dim = args.num_past - 1
    else:
        edge_attr = dataset[0][1]
        in_edge_nf = int(edge_attr.shape[-1])
        nodes_att_dim = 0

    model = ESTAG(
        num_past=args.num_past,
        num_future=1,
        in_node_nf=args.in_node_nf,
        in_edge_nf=in_edge_nf,
        hidden_nf=args.hidden_nf,
        fft=args.fft,
        eat=args.eat,
        device=DEVICE,
        n_layers=args.n_layers,
        n_nodes=dataset.n_node,
        nodes_att_dim=nodes_att_dim,
    ).to(DEVICE)

    print("Architecture :")
    print(f"  in_node_nf   = {args.in_node_nf}")
    print(f"  in_edge_nf   = {in_edge_nf}")
    print(f"  hidden_nf    = {args.hidden_nf}")
    print(f"  n_layers     = {args.n_layers}")
    print(f"  fft          = {args.fft}")
    print(f"  eat          = {args.eat}")
    print(f"  nodes_att_dim= {nodes_att_dim}")

    return model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Le checkpoint chargé n'est pas un dictionnaire.")

    for key in ("state_dict", "model_state_dict", "model", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state: dict[str, torch.Tensor] = {}

    for name, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        clean_name = name
        for prefix in ("module.", "_orig_mod."):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]

        state[clean_name] = value

    if not state:
        raise ValueError("Aucun tenseur de poids trouvé dans le checkpoint.")

    return state


def load_checkpoint(model: ESTAG, path: Path, strict: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    state = extract_state_dict(checkpoint)

    try:
        incompatible = model.load_state_dict(state, strict=strict)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint incompatible : {path}\n"
            "Vérifie --fft/--no-fft, --hidden-nf, --n-layers et --num-past.\n\n"
            f"Erreur PyTorch :\n{error}"
        ) from error

    if not strict:
        print("Clés manquantes  :", incompatible.missing_keys)
        print("Clés inattendues:", incompatible.unexpected_keys)


@torch.no_grad()
def evaluate_rollouts(
    model: ESTAG,
    dataset: MD17Dataset,
    loader: DataLoader,
    n_rollouts: int,
    max_steps: int,
    num_past: int,
    delta_frame: int,
) -> np.ndarray:
    model.eval()
    n_nodes = dataset.n_node

    mse_matrix = np.full(
        (n_rollouts, max_steps), np.nan, dtype=np.float64
    )

    row_offset = 0

    for batch_number, batch in enumerate(loader, start=1):
        (
            current_window,
            edge_attr_batch,
            node_feat,
            start_idx,
            edges,
            batch_size,
        ) = prepare_batch(batch, dataset, num_past)

        row_slice = slice(row_offset, row_offset + batch_size)

        for step in range(1, max_steps + 1):
            pred = model(node_feat, current_window, edges, edge_attr_batch)

            target_indices = start_idx + delta_frame * (
                num_past + step - 1
            )
            target = get_ground_truth(dataset, target_indices).reshape(
                batch_size * n_nodes, 3
            )

            per_sample_mse = (
                (pred - target)
                .reshape(batch_size, n_nodes, 3)
                .pow(2)
                .mean(dim=(1, 2))
            )

            values = (
                per_sample_mse.detach().cpu().numpy().astype(np.float64)
            )
            values[~np.isfinite(values)] = np.nan
            mse_matrix[row_slice, step - 1] = values

            # Rollout brut : aucune correction et aucun Kabsch.
            current_window = torch.cat(
                [current_window[1:], pred.unsqueeze(0)], dim=0
            )

            if step == 1 or step % 50 == 0 or step == max_steps:
                n_finite = int(np.isfinite(values).sum())
                print(
                    f"batch {batch_number:02d} | pas {step:03d}/{max_steps} | "
                    f"finis : {n_finite}/{batch_size}"
                )

        row_offset += batch_size

    if row_offset != n_rollouts:
        raise RuntimeError(
            f"{row_offset} rollouts traités au lieu de {n_rollouts}."
        )

    return mse_matrix


def save_matrix_outputs(
    matrix: np.ndarray,
    absolute_starts: list[int],
    output_dir: Path,
    prefix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / f"{prefix}_matrix.csv"
    matrix_npy = output_dir / f"{prefix}.npy"
    detailed_csv = output_dir / f"{prefix}_matrix_with_starts.csv"
    by_step_csv = output_dir / f"{prefix}_by_step.csv"

    # Même format que les matrices existantes : sans en-tête.
    np.savetxt(matrix_csv, matrix, delimiter=",", fmt="%.10e")
    np.save(matrix_npy, matrix)

    with detailed_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["start_idx"]
            + [f"step_{step}" for step in range(1, matrix.shape[1] + 1)]
        )
        for start, row in zip(absolute_starts, matrix):
            writer.writerow([start, *row])

    with np.errstate(invalid="ignore"):
        means = np.nanmean(matrix, axis=0)
        medians = np.nanmedian(matrix, axis=0)

    n_finite = np.isfinite(matrix).sum(axis=0)
    n_nan = matrix.shape[0] - n_finite

    with by_step_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "mean_mse", "median_mse", "n_finite", "n_nan"])
        for step in range(1, matrix.shape[1] + 1):
            writer.writerow(
                [
                    step,
                    means[step - 1],
                    medians[step - 1],
                    int(n_finite[step - 1]),
                    int(n_nan[step - 1]),
                ]
            )

    return matrix_csv, matrix_npy, detailed_csv, by_step_csv, means


def detect_csv_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except csv.Error:
        return ","


def update_summary_csv(path: Path, column_name: str, means: np.ndarray) -> None:
    mean_by_step = {
        step: float(means[step - 1]) for step in range(1, len(means) + 1)
    }

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["step", column_name])
            writer.writeheader()
            for step, value in mean_by_step.items():
                writer.writerow({"step": step, column_name: value})
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)

    delimiter = detect_csv_delimiter(path)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise ValueError(f"Le CSV résumé ne contient pas d'en-tête : {path}")

    step_column = next(
        (
            candidate
            for candidate in ("step", "Step", "pas", "prediction")
            if candidate in fieldnames
        ),
        None,
    )

    if step_column is None:
        raise ValueError(
            f"Colonne de pas introuvable dans {path}. "
            f"Colonnes disponibles : {fieldnames}"
        )

    if column_name not in fieldnames:
        fieldnames.append(column_name)

    existing_steps: set[int] = set()

    for row in rows:
        try:
            step = int(float(row[step_column]))
        except (TypeError, ValueError):
            continue

        existing_steps.add(step)
        if step in mean_by_step:
            row[column_name] = mean_by_step[step]

    for step in range(1, len(means) + 1):
        if step not in existing_steps:
            row = {field: "" for field in fieldnames}
            row[step_column] = step
            row[column_name] = mean_by_step[step]
            rows.append(row)

    rows.sort(key=lambda row: int(float(row[step_column])))

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sauvegarde de sécurité : {backup_path}")


def print_selected_steps(matrix: np.ndarray) -> None:
    print("\nRésumé :")
    for step in (1, 11, 100, 200):
        if step > matrix.shape[1]:
            continue
        values = matrix[:, step - 1]
        finite = values[np.isfinite(values)]
        mean = float(np.mean(finite)) if len(finite) else float("nan")
        print(
            f"  pas {step:3d} : MSE moyenne = {mean:.8e} | "
            f"finis = {len(finite)}/{len(values)}"
        )


def main() -> None:
    args = parse_args()

    print("Périphérique :", DEVICE)
    print("Checkpoint   :", args.checkpoint)
    print("Départs      :", args.fixed_starts)
    print("Nombre de pas:", args.max_steps)

    fixed_values = read_fixed_starts(args.fixed_starts)
    print(f"{len(fixed_values)} départs lus")

    dataset = MD17Dataset(
        partition=args.partition,
        max_samples=args.max_samples,
        data_dir=args.data_dir,
        molecule_type=args.molecule,
        delta_frame=args.delta_frame,
        num_past=args.num_past,
    )

    subset_indices, absolute_starts, interpretation = resolve_subset_indices(
        dataset=dataset,
        fixed_values=fixed_values,
        max_steps=args.max_steps,
        num_past=args.num_past,
        delta_frame=args.delta_frame,
    )

    print("Interprétation du fichier :", interpretation)
    print("Premier start_idx absolu  :", absolute_starts[0])
    print("Dernier start_idx absolu  :", absolute_starts[-1])

    subset = Subset(dataset, subset_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = build_model(dataset, args)
    load_checkpoint(model, args.checkpoint, strict=not args.non_strict)

    matrix = evaluate_rollouts(
        model=model,
        dataset=dataset,
        loader=loader,
        n_rollouts=len(subset_indices),
        max_steps=args.max_steps,
        num_past=args.num_past,
        delta_frame=args.delta_frame,
    )

    matrix_csv, matrix_npy, detailed_csv, by_step_csv, means = (
        save_matrix_outputs(
            matrix=matrix,
            absolute_starts=absolute_starts,
            output_dir=args.output_dir,
            prefix=args.output_prefix,
        )
    )

    if not args.no_update_summary:
        update_summary_csv(
            path=args.summary_csv,
            column_name=args.summary_column,
            means=means,
        )

    print_selected_steps(matrix)

    print("\nFichiers créés :")
    print(" ", matrix_csv)
    print(" ", matrix_npy)
    print(" ", detailed_csv)
    print(" ", by_step_csv)

    if not args.no_update_summary:
        print(" ", args.summary_csv)
        print("Colonne ajoutée :", args.summary_column)


if __name__ == "__main__":
    main()
