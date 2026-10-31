# eval_estag_rollout_mse.py

import os
import csv
import argparse
import numpy as np
import torch

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# UTILS
# ============================================================






def load_state_dict(path, device):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # enlève éventuellement "module."
    clean = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v

    return clean


def infer_fft_from_checkpoint(state_dict, num_past):
    """
    Détecte fft=True/False à partir des shapes du checkpoint.
    Si adj_mlp attend num_past-1 features, alors fft=True.
    Si adj_mlp attend 3 features, alors fft=False.
    """

    for name, w in state_dict.items():

        if name.endswith("adj_mlp.0.weight") and w.ndim == 2:

            dim = w.shape[1]

            if dim == num_past - 1:
                return True

            if dim == 3:
                return False

    return None


def get_start_idx(dataset, sample, i):
    """
    Ton dataset.py peut renvoyer start_idx dans __getitem__.
    Sinon on essaie dataset.st[i].
    """

    if len(sample) >= 5:
        start_idx = sample[4]

        if torch.is_tensor(start_idx):
            return int(start_idx.item())

        return int(start_idx)

    if hasattr(dataset, "st"):
        return int(dataset.st[i])

    raise ValueError(
        "Impossible de récupérer start_idx. "
        "Il faut que MD17Dataset.__getitem__ renvoie start_idx "
        "ou que dataset.st existe."
    )


def unpack_sample(sample):
    """
    Accepte :
        loc, edge_attr, node_feat, loc_end
    ou :
        loc, edge_attr, node_feat, loc_end, start_idx
    """

    if len(sample) == 4:
        loc, edge_attr, node_feat, loc_end = sample

    elif len(sample) >= 5:
        loc, edge_attr, node_feat, loc_end, _ = sample

    else:
        raise ValueError(
            f"Sample de longueur inattendue : {len(sample)}"
        )

    return loc, edge_attr, node_feat, loc_end


def as_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)

    return torch.as_tensor(
        x,
        dtype=torch.float32,
        device=device,
    )


def get_frame(dataset, frame_idx, device):
    """
    Récupère dataset.x[frame_idx] sous forme torch.Tensor (N,3).
    """

    x = dataset.x

    if torch.is_tensor(x):
        frame = x[frame_idx].to(
            device=device,
            dtype=torch.float32,
        )

    else:
        frame = torch.as_tensor(
            x[frame_idx],
            dtype=torch.float32,
            device=device,
        )

    return frame


def valid_start(dataset, start_idx, num_past, delta_frame, max_step):
    """
    step=1 correspond à :
        start + delta_frame * num_past

    step=s correspond à :
        start + delta_frame * (num_past + s - 1)
    """

    last_idx = start_idx + delta_frame * (
        num_past + max_step - 1
    )

    return last_idx < len(dataset.x)


# ============================================================
# EVAL ROLLOUT
# ============================================================

@torch.no_grad()
def eval_rollout_from_sample(
    model,
    dataset,
    sample,
    start_idx,
    eval_steps,
    device,
    num_past,
    delta_frame,
):
    """
    Fait un rollout depuis un sample test.

    Retourne :
        dict {step: mse}
    """

    n_nodes = dataset.n_node
    max_step = max(eval_steps)

    loc, edge_attr, node_feat, _ = unpack_sample(sample)

    # loc : (T,N,3)
    current = as_tensor(
        loc,
        device,
    ).reshape(num_past, n_nodes, 3)

    edge_attr = as_tensor(
        edge_attr,
        device,
    )

    node_feat = as_tensor(
        node_feat,
        device,
    ).reshape(n_nodes, -1)

    edges = dataset.get_edges(
        batch_size=1,
        n_nodes=n_nodes,
    )

    edges = [
        e.to(device)
        for e in edges
    ]

    mses = {}

    for step in range(1, max_step + 1):

        pred = model(
            node_feat,
            current,
            edges,
            edge_attr,
        )

        target_idx = start_idx + delta_frame * (
            num_past + step - 1
        )

        target = get_frame(
            dataset,
            target_idx,
            device,
        ).reshape(n_nodes, 3)

        mse = torch.mean(
            (pred - target) ** 2
        ).item()

        if step in eval_steps:
            mses[step] = mse

        current = torch.cat(
            [
                current[1:],
                pred.unsqueeze(0),
            ],
            dim=0,
        )

    return mses

def read_starts_file(filename):
    starts = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            starts.append(int(line.split(",")[0]))

    return starts


def write_starts_file(filename, starts):
    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    with open(filename, "w") as f:
        f.write("# start_idx\n")

        for s in starts:
            f.write(f"{int(s)}\n")


def select_fixed_starts(
    dataset,
    num_starts,
    num_past,
    delta_frame,
    max_eval_step,
    starts_file=None,
    overwrite_starts=False,
):
    """
    Retourne une liste :
        [(dataset_i, start_idx), ...]

    Si starts_file existe, on utilise ces start_idx.
    Sinon on sélectionne les premiers start_idx valides et on les sauvegarde.
    """

    # Dictionnaire start_idx -> dataset_i
    start_to_i = {}

    for i in range(len(dataset)):
        sample = dataset[i]
        start_idx = get_start_idx(dataset, sample, i)
        start_to_i[int(start_idx)] = i

    # --------------------------------------------------------
    # Cas 1 : fichier existant
    # --------------------------------------------------------
    if (
        starts_file is not None
        and os.path.isfile(starts_file)
        and not overwrite_starts
    ):
        fixed_starts = read_starts_file(starts_file)

        selected = []

        for start_idx in fixed_starts:

            if start_idx not in start_to_i:
                print(
                    f"[warning] start_idx={start_idx} absent du dataset test actuel, ignoré."
                )
                continue

            if not valid_start(
                dataset,
                start_idx,
                num_past,
                delta_frame,
                max_eval_step,
            ):
                print(
                    f"[warning] start_idx={start_idx} pas assez long pour step={max_eval_step}, ignoré."
                )
                continue

            selected.append(
                (
                    start_to_i[start_idx],
                    start_idx,
                )
            )

        if len(selected) == 0:
            raise RuntimeError(
                f"Aucun start_idx valide trouvé dans {starts_file}"
            )

        return selected[:num_starts]

    # --------------------------------------------------------
    # Cas 2 : génération
    # --------------------------------------------------------
    selected = []

    for i in range(len(dataset)):

        sample = dataset[i]
        start_idx = get_start_idx(dataset, sample, i)

        if valid_start(
            dataset,
            start_idx,
            num_past,
            delta_frame,
            max_eval_step,
        ):
            selected.append(
                (
                    i,
                    int(start_idx),
                )
            )

        if len(selected) >= num_starts:
            break

    if len(selected) == 0:
        raise RuntimeError(
            "Aucun départ valide trouvé."
        )

    if starts_file is not None:
        write_starts_file(
            starts_file,
            [s for _, s in selected],
        )

        print("start_idx sauvegardés dans :", starts_file)

    return selected



# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default = "resultats/transfer_aspirin_uracil/entrainement_juste_sur_uracil.pt",
        help="Chemin du checkpoint .pt à évaluer",
    )

    parser.add_argument(
        "--mol",
        default="aspirin",
        help="Molécule MD17 : aspirin, uracil, salicylic, etc.",
    )

    parser.add_argument(
        "--data_dir",
        default="md17",
    )

    parser.add_argument(
        "--result_dir",
        default="resultats/eval_rollout_mse",
    )

    parser.add_argument(
        "--num_starts",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num_past",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--delta_frame",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval_steps",
        type=str,
        default="1,11,100,200,500",
        help="Steps de rollout à évaluer, séparés par des virgules",
    )

    parser.add_argument(
        "--in_node_nf",
        type=int,
        default=513,
    )

    parser.add_argument(
        "--hidden_nf",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--fft",
        choices=["auto", "true", "false"],
        default="auto",
    )

    parser.add_argument(
        "--starts_file",
        type=str,
        default=None,
        help="Fichier contenant les start_idx fixes à utiliser. S'il n'existe pas, il sera créé.",
    )

    parser.add_argument(
        "--overwrite_starts",
        action="store_true",
        help="Force la regénération du fichier de start_idx.",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(
        args.result_dir,
        exist_ok=True,
    )

    eval_steps = [
        int(x)
        for x in args.eval_steps.split(",")
    ]

    max_eval_step = max(eval_steps)

    print("device:", device)
    print("checkpoint:", args.checkpoint)
    print("mol:", args.mol)
    print("eval_steps:", eval_steps)

    # ------------------------------------------------------------
    # Dataset test
    # ------------------------------------------------------------

    dataset = MD17Dataset(
        partition="test",
        max_samples=5000,
        data_dir=args.data_dir,
        molecule_type=args.mol,
        delta_frame=args.delta_frame,
        num_past=args.num_past,
    )

    n_nodes = dataset.n_node

    print("n_nodes:", n_nodes)
    print("test samples:", len(dataset))

    # ------------------------------------------------------------
    # Checkpoint + FFT detection
    # ------------------------------------------------------------

    state_dict = load_state_dict(
        args.checkpoint,
        device,
    )

    inferred_fft = infer_fft_from_checkpoint(
        state_dict,
        args.num_past,
    )

    if args.fft == "auto":

        if inferred_fft is None:
            raise ValueError(
                "Impossible d'inférer fft depuis le checkpoint. "
                "Passe explicitement --fft true ou --fft false."
            )

        fft = inferred_fft

    else:

        fft = args.fft == "true"

    if fft:
        in_edge_nf = args.num_past - 1
        nodes_att_dim = args.num_past - 1
    else:
        in_edge_nf = 3
        nodes_att_dim = 0

    print("fft:", fft)
    print("in_edge_nf:", in_edge_nf)
    print("nodes_att_dim:", nodes_att_dim)

    # ------------------------------------------------------------
    # Modèle
    # ------------------------------------------------------------

    model = ESTAG(
        num_past=args.num_past,
        num_future=1,
        in_node_nf=args.in_node_nf,
        in_edge_nf=in_edge_nf,
        hidden_nf=args.hidden_nf,
        fft=fft,
        eat=True,
        nodes_att_dim=nodes_att_dim,
        device=device,
        n_layers=args.n_layers,
        n_nodes=n_nodes,
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print("model loaded")

    # ------------------------------------------------------------
    # Choix des départs de rollout
    # ------------------------------------------------------------

    
    if args.starts_file is None:
        args.starts_file = os.path.join(
            args.result_dir,
            f"fixed_starts_{args.mol}_step_{max_eval_step}.txt",
        )

    selected = select_fixed_starts(
        dataset=dataset,
        num_starts=args.num_starts,
        num_past=args.num_past,
        delta_frame=args.delta_frame,
        max_eval_step=max_eval_step,
        starts_file=args.starts_file,
        overwrite_starts=args.overwrite_starts,
    )

    print("rollout starts selected:", len(selected))
    print("starts_file:", args.starts_file)

    # selected = []

    # for i in range(len(dataset)):

    #     sample = dataset[i]

    #     start_idx = get_start_idx(
    #         dataset,
    #         sample,
    #         i,
    #     )

    #     if valid_start(
    #         dataset,
    #         start_idx,
    #         args.num_past,
    #         args.delta_frame,
    #         max_eval_step,
    #     ):

    #         selected.append(
    #             (i, start_idx)
    #         )

    #     if len(selected) >= args.num_starts:
    #         break

    # if len(selected) == 0:
    #     raise RuntimeError(
    #         "Aucun départ valide trouvé pour ce max_eval_step."
    #     )

    # print("rollout starts selected:", len(selected))

    # ------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------

    per_start_rows = []

    sums = {
        step: 0.0
        for step in eval_steps
    }

    counts = {
        step: 0
        for step in eval_steps
    }

    for local_id, (dataset_i, start_idx) in enumerate(selected):

        sample = dataset[dataset_i]

        mses = eval_rollout_from_sample(
            model=model,
            dataset=dataset,
            sample=sample,
            start_idx=start_idx,
            eval_steps=eval_steps,
            device=device,
            num_past=args.num_past,
            delta_frame=args.delta_frame,
        )

        row = {
            "local_id": local_id,
            "dataset_i": dataset_i,
            "start_idx": start_idx,
        }

        for step in eval_steps:

            value = mses[step]

            row[f"mse_step_{step}"] = value

            sums[step] += value
            counts[step] += 1

        per_start_rows.append(row)

        print(
            f"[{local_id+1:02d}/{len(selected):02d}] "
            f"dataset_i={dataset_i} start={start_idx} | "
            + " ".join(
                [
                    f"s{step}={mses[step]:.8f}"
                    for step in eval_steps
                ]
            )
        )

    mean_mses = {
        step: sums[step] / counts[step]
        for step in eval_steps
    }

    # ------------------------------------------------------------
    # Logs CSV
    # ------------------------------------------------------------

    checkpoint_name = "ESTAG_vanilla" #os.path.splitext(
    #     os.path.basename(args.checkpoint)
    # )[0]

    per_start_file = os.path.join(
        args.result_dir,
        f"{checkpoint_name}_{args.mol}_per_start.csv",
    )

    summary_file = os.path.join(
        args.result_dir,
        f"{checkpoint_name}_{args.mol}_summary.csv",
    )

    # per-start log
    fieldnames = [
        "local_id",
        "dataset_i",
        "start_idx",
    ] + [
        f"mse_step_{step}"
        for step in eval_steps
    ]

    with open(per_start_file, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in per_start_rows:
            writer.writerow(row)

    # summary log
    with open(summary_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "checkpoint",
                "molecule",
                "num_starts",
                "num_past",
                "delta_frame",
                "fft",
            ] + [
                f"mean_mse_step_{step}"
                for step in eval_steps
            ]
        )

        writer.writerow(
            [
                args.checkpoint,
                args.mol,
                len(selected),
                args.num_past,
                args.delta_frame,
                fft,
            ] + [
                mean_mses[step]
                for step in eval_steps
            ]
        )

    print()
    print("=== MSE moyennes ===")
    for step in eval_steps:
        print(f"step {step:4d} : {mean_mses[step]:.10f}")

    print()
    print("saved per-start log:", per_start_file)
    print("saved summary log  :", summary_file)


if __name__ == "__main__":
    main()