# eval_rollout_curves.py

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import mlatom as ml

from md17.dataset import MD17Dataset
from models.model import ESTAG as ESTAGClassic
from models.model_unimol import ESTAG as ESTAGUnimol


PTABLE = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P", 16:"S", 17:"Cl"}


def load_state(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def infer_hidden_nf(state):
    if "TimeEmbedding.weight" in state:
        return state["TimeEmbedding.weight"].shape[1]
    for k, v in state.items():
        if k.endswith("edge_mlp.0.bias"):
            return v.shape[0]
    raise ValueError("Impossible d'inférer hidden_nf.")


def infer_fft(state, num_past):
    for k, v in state.items():
        if k.endswith("adj_mlp.0.weight") and v.ndim == 2:
            if v.shape[1] == num_past - 1:
                return True
            if v.shape[1] == 3:
                return False
    raise ValueError("Impossible d'inférer fft.")


def get_symbols(dataset):
    if hasattr(dataset, "z"):
        z = dataset.z
    elif hasattr(dataset, "Z"):
        z = dataset.Z
    elif hasattr(dataset, "mole_idx"):
        z = dataset.mole_idx
    else:
        z = dataset.node_feat[:, 0]

    z = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    return [PTABLE.get(int(round(float(a))), "X") for a in z]


def get_frame(dataset, frame_id):
    x = dataset.x
    if torch.is_tensor(x):
        return x[frame_id].detach().cpu().numpy().astype(np.float32)
    return np.asarray(x[frame_id], dtype=np.float32)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def compute_velocities(x, dt_fs):
    v = np.empty_like(x, dtype=np.float32)
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt_fs)
    v[0] = (x[1] - x[0]) / dt_fs
    v[-1] = (x[-1] - x[-2]) / dt_fs
    return v


def read_starts(path):
    with open(path) as f:
        return [int(line.strip().split(",")[0]) for line in f if line.strip() and not line.startswith("#")]


def write_starts(path, starts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# start_idx\n")
        for s in starts:
            f.write(f"{int(s)}\n")


def select_starts(dataset, args):
    max_target_offset = args.delta_frame * (args.num_past + args.max_steps - 1)
    if args.starts_file and os.path.isfile(args.starts_file) and not args.overwrite_starts:
        starts = read_starts(args.starts_file)
        starts = [s for s in starts if s + max_target_offset < len(dataset.x)]
        if len(starts) == 0:
            raise RuntimeError("Aucun start_idx valide dans starts_file.")
        return starts[:args.num_starts]

    candidates = list(dataset.st) if hasattr(dataset, "st") else list(range(len(dataset.x)))
    starts = [int(s) for s in candidates if int(s) + max_target_offset < len(dataset.x)]
    starts = starts[:args.num_starts]

    if len(starts) == 0:
        raise RuntimeError("Aucun start_idx valide trouvé.")

    if args.starts_file:
        write_starts(args.starts_file, starts)

    return starts


def prepare_dataset_inputs(dataset, device):
    sample = dataset[0]
    if len(sample) == 5:
        _, edge_attr, node_feat, _, _ = sample
    else:
        _, edge_attr, node_feat, _ = sample

    edges = dataset.get_edges(1, dataset.n_node)
    node_feat = node_feat.to(device, dtype=torch.float32).reshape(dataset.n_node, -1)

    return {
        "edge_attr": edge_attr.to(device),
        "edges": [e.to(device) for e in edges],
        "node_feat_classic": node_feat[:, :1],
        "node_feat_unimol": node_feat,
    }


def build_estag(cfg, dataset, edge_attr, device, num_past):
    state = load_state(cfg["checkpoint"], device)
    hidden_nf = infer_hidden_nf(state) if cfg["hidden_nf"] is None else cfg["hidden_nf"]
    fft = infer_fft(state, num_past) if cfg["fft"] == "auto" else bool(cfg["fft"])

    model = cfg["class"](
        num_past=num_past, num_future=1, in_node_nf=cfg["in_node_nf"],
        in_edge_nf=num_past - 1 if fft else edge_attr.shape[-1],
        hidden_nf=hidden_nf, fft=fft, eat=True,
        nodes_att_dim=num_past - 1 if fft else 0,
        device=device, n_layers=cfg["n_layers"], n_nodes=dataset.n_node,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"{cfg['name']} chargé | hidden_nf={hidden_nf} | fft={fft} | in_node_nf={cfg['in_node_nf']}")
    return model


@torch.no_grad()
def rollout_estag(model, feat, inputs, dataset, start_idx, args, device):
    ids = start_idx + args.delta_frame * np.arange(args.num_past)
    frames0 = np.asarray([get_frame(dataset, int(i)) for i in ids], dtype=np.float32)
    current = torch.tensor(frames0, dtype=torch.float32, device=device)
    out = np.full(args.max_steps, np.nan, dtype=np.float64)

    for step in range(1, args.max_steps + 1):
        pred = model(feat, current, inputs["edges"], inputs["edge_attr"])
        pred_np = pred.detach().cpu().numpy()
        target_id = start_idx + args.delta_frame * (args.num_past + step - 1)
        out[step - 1] = mse(pred_np, get_frame(dataset, target_id))
        current = torch.cat([current[1:], pred.unsqueeze(0)], dim=0)

        if not np.isfinite(out[step - 1]):
            break

    return out


def make_mlatom_mol(symbols, coords, velocities):
    mol = ml.data.molecule.from_numpy(coordinates=np.asarray(coords, dtype=np.float32), species=np.asarray(symbols))
    mol.add_xyz_vectorial_property(np.asarray(velocities, dtype=np.float32), "xyz_velocities")
    return mol


def get_db_coords(db, step, max_steps):
    if len(db) >= max_steps + 1:
        return np.asarray(db[step].xyz_coordinates, dtype=np.float32)
    return np.asarray(db[step - 1].xyz_coordinates, dtype=np.float32)


def rollout_mdtrajnet(model, ani1ccx, symbols, velocities, dataset, start_idx, args):
    last_input_id = start_idx + args.delta_frame * (args.num_past - 1)
    mol0 = make_mlatom_mol(symbols, get_frame(dataset, last_input_id), velocities[last_input_id])

    out = np.full(args.max_steps, np.nan, dtype=np.float64)

    try:
        traj = model.propagate(
            molecule=mol0,
            time=args.max_steps * args.estag_dt_fs,
            time_step=args.estag_dt_fs,
            time_segment=2 * args.estag_dt_fs,
            rescale_velocities=args.mdtrajnet_rescale,
            potential_model=ani1ccx if args.mdtrajnet_rescale else None,
        ).to_database()
    except Exception as e:
        print(f"[warning] MDTrajNet échec start_idx={start_idx}: {e}")
        return out

    for step in range(1, args.max_steps + 1):
        try:
            pred = get_db_coords(traj, step, args.max_steps)
            target_id = start_idx + args.delta_frame * (args.num_past + step - 1)
            out[step - 1] = mse(pred, get_frame(dataset, target_id))
        except Exception:
            out[step - 1] = np.nan

    return out


def save_matrix_csv(path, starts, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_idx"] + [f"step_{i}" for i in range(1, arr.shape[1] + 1)])
        for s, row in zip(starts, arr):
            w.writerow([s] + list(row))


def save_summary(path, steps, curves):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "time_fs"]
        for name in curves:
            header += [f"{name}_mean", f"{name}_n_valid"]
        w.writerow(header)

        for i, step in enumerate(steps):
            row = [step, step * ARGS.estag_dt_fs]
            for name, arr in curves.items():
                vals = arr[:, i]
                good = np.isfinite(vals)
                row += [np.nanmean(vals) if np.any(good) else np.nan, int(np.sum(good))]
            w.writerow(row)


def plot_curves(path, curves, plot_until, estag_dt_fs):
    steps = np.arange(1, plot_until + 1)
    plt.figure(figsize=(8, 5), dpi=160)

    for name, arr in curves.items():
        y = np.nanmean(arr[:, :plot_until], axis=0)
        ok = np.isfinite(y) & (y > 0)
        plt.plot(steps[ok], y[ok], label=name)

    plt.yscale("log")
    plt.xlabel("step ESTAG")
    plt.ylabel("MSE moyenne")
    plt.title(f"MSE rollout aspirin test, jusqu'à {plot_until} steps ({plot_until * estag_dt_fs:.0f} fs)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="md17")
    p.add_argument("--mol", default="aspirin")
    p.add_argument("--result_dir", default="resultats/rollout_courbes_aspirine_rapport/")
    p.add_argument("--classic_ckpt", default="logs/md17_logs/exp_aspirin/estag/saved_classical_model_weights.pt")
    p.add_argument("--unimol_ckpt", default="resultats/estag_unimol_rollout_fft_aspirine.pt")
    p.add_argument("--mdtrajnet_model", default="MDtrajNet1")
    p.add_argument("--num_starts", type=int, default=100)
    p.add_argument("--dataset_max_samples", type=int, default=10000)
    p.add_argument("--num_past", type=int, default=10)
    p.add_argument("--delta_frame", type=int, default=10)
    p.add_argument("--md17_dt_fs", type=float, default=0.5)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--plot_until", type=int, default=220)
    p.add_argument("--starts_file", default=None)
    p.add_argument("--overwrite_starts", action="store_true")
    p.add_argument("--no_mdtrajnet_rescale", dest="mdtrajnet_rescale", action="store_false")
    p.set_defaults(mdtrajnet_rescale=True)
    ARGS = p.parse_args()

    ARGS.estag_dt_fs = ARGS.delta_frame * ARGS.md17_dt_fs
    ARGS.plot_until = min(ARGS.plot_until, ARGS.max_steps)
    os.makedirs(ARGS.result_dir, exist_ok=True)

    if ARGS.starts_file is None:
        ARGS.starts_file = os.path.join(ARGS.result_dir, f"fixed_starts_{ARGS.mol}_{ARGS.num_starts}_max{ARGS.max_steps}.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MD17Dataset(
        partition="test", max_samples=max(ARGS.dataset_max_samples, ARGS.num_starts),
        data_dir=ARGS.data_dir, molecule_type=ARGS.mol,
        delta_frame=ARGS.delta_frame, num_past=ARGS.num_past,
    )

    starts = select_starts(dataset, ARGS)
    symbols = get_symbols(dataset)
    inputs = prepare_dataset_inputs(dataset, device)
    velocities = compute_velocities(np.asarray(dataset.x, dtype=np.float32), ARGS.md17_dt_fs)

    print("device:", device)
    print("starts:", len(starts))
    print("starts_file:", ARGS.starts_file)
    print("dt ESTAG:", ARGS.estag_dt_fs, "fs")
    print("max_steps:", ARGS.max_steps)

    configs = [
        {"name":"estag_classic", "checkpoint":ARGS.classic_ckpt, "class":ESTAGClassic, "in_node_nf":1, "hidden_nf":None, "fft":"auto", "n_layers":2},
        {"name":"estag_unimol", "checkpoint":ARGS.unimol_ckpt, "class":ESTAGUnimol, "in_node_nf":513, "hidden_nf":None, "fft":"auto", "n_layers":2},
    ]

    curves = {}

    for cfg in configs:
        if not os.path.isfile(cfg["checkpoint"]):
            print("[skip]", cfg["name"], "checkpoint introuvable:", cfg["checkpoint"])
            continue

        model = build_estag(cfg, dataset, inputs["edge_attr"], device, ARGS.num_past)
        feat = inputs["node_feat_classic"] if cfg["in_node_nf"] == 1 else inputs["node_feat_unimol"]
        arr = np.full((len(starts), ARGS.max_steps), np.nan, dtype=np.float64)

        for j, s in enumerate(starts):
            arr[j] = rollout_estag(model, feat, inputs, dataset, s, ARGS, device)
            print(f"{cfg['name']} | {j+1:03d}/{len(starts):03d}")

        curves[cfg["name"]] = arr
        np.save(os.path.join(ARGS.result_dir, f"mse_{cfg['name']}.npy"), arr)
        save_matrix_csv(os.path.join(ARGS.result_dir, f"mse_{cfg['name']}_matrix.csv"), starts, arr)

    try:
        mdtrajnet = ml.MDtrajNet(model_file=ARGS.mdtrajnet_model, verbose=False)
        ani1ccx = ml.methods("ANI-1ccx") if ARGS.mdtrajnet_rescale else None
        arr = np.full((len(starts), ARGS.max_steps), np.nan, dtype=np.float64)

        for j, s in enumerate(starts):
            arr[j] = rollout_mdtrajnet(mdtrajnet, ani1ccx, symbols, velocities, dataset, s, ARGS)
            print(f"mdtrajnet | {j+1:03d}/{len(starts):03d}")

        curves["mdtrajnet"] = arr
        np.save(os.path.join(ARGS.result_dir, "mse_mdtrajnet.npy"), arr)
        save_matrix_csv(os.path.join(ARGS.result_dir, "mse_mdtrajnet_matrix.csv"), starts, arr)

    except Exception as e:
        print("[skip] mdtrajnet:", e)

    steps = list(range(1, ARGS.max_steps + 1))
    summary_path = os.path.join(ARGS.result_dir, "mse_summary_by_step.csv")
    plot_path = os.path.join(ARGS.result_dir, f"mse_curves_log_until_{ARGS.plot_until}.png")

    save_summary(summary_path, steps, curves)
    plot_curves(plot_path, curves, ARGS.plot_until, ARGS.estag_dt_fs)

    print()
    print("Résumé CSV :", summary_path)
    print("Graphique  :", plot_path)

    for name, arr in curves.items():
        mean = np.nanmean(arr, axis=0)
        for s in [1, 11, 100, 200, 500, 1000]:
            if s <= ARGS.max_steps:
                vals = arr[:, s - 1]
                print(f"{name:15s} step={s:4d} t={s*ARGS.estag_dt_fs:8.2f} fs mean={mean[s-1]:.8f} n={np.isfinite(vals).sum()}")


if __name__ == "__main__":
    main()