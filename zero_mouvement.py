# zero_mouvement_compare.py

import os, csv, argparse

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np
import torch
import mlatom as ml

from md17.dataset import MD17Dataset
from models.model import ESTAG as ESTAGClassic
from models.model_unimol import ESTAG as ESTAGUnimol
from unimol_tools import UniMolRepr


PTABLE = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P", 16:"S", 17:"Cl"}
ZTABLE = {v:k for k, v in PTABLE.items()}


# ============================================================
# IO / UTILS
# ============================================================

def read_xyz_one_frame(path):
    with open(path) as f:
        n = int(f.readline())
        f.readline()
        symbols, coords = [], []
        for _ in range(n):
            p = f.readline().split()
            symbols.append(p[0])
            coords.append([float(p[1]), float(p[2]), float(p[3])])
    return symbols, np.asarray(coords, dtype=np.float32)


def write_xyz(path, symbols, frames, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i, frame in enumerate(frames):
            f.write(f"{len(symbols)}\n{name} zero_mouvement step={i}\n")
            for s, r in zip(symbols, frame):
                f.write(f"{s:2s} {r[0]:16.8f} {r[1]:16.8f} {r[2]:16.8f}\n")


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
    raise ValueError("Impossible d'inférer hidden_nf depuis le checkpoint.")


def infer_fft(state, num_past):
    for k, v in state.items():
        if k.endswith("adj_mlp.0.weight") and v.ndim == 2:
            if v.shape[1] == num_past - 1:
                return True
            if v.shape[1] == 3:
                return False
    raise ValueError("Impossible d'inférer fft depuis le checkpoint.")


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def symbols_to_z(symbols):
    return np.asarray([ZTABLE[s] for s in symbols], dtype=np.float32)


def get_dataset_symbols(dataset):
    if hasattr(dataset, "z"):
        z = dataset.z
    elif hasattr(dataset, "Z"):
        z = dataset.Z
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

    return [PTABLE.get(int(round(float(x))), "X") for x in z]


def get_frame(dataset, frame_id):
    x = dataset.x
    if torch.is_tensor(x):
        return x[frame_id].detach().cpu().numpy().astype(np.float32)
    return np.asarray(x[frame_id], dtype=np.float32)


# ============================================================
# GRAPH / FEATURES
# ============================================================

def build_graph_from_coords(symbols, coords, cutoff=1.6):
    z = symbols_to_z(symbols)
    n = len(symbols)
    atom_edge = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(coords[i] - coords[j]) < cutoff:
                atom_edge[i, j] = 1

    atom_edge2 = atom_edge @ atom_edge
    rows, cols, attrs = [], [], []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if atom_edge[i, j]:
                rows.append(i); cols.append(j); attrs.append([z[i], z[j], 1.0])
            elif atom_edge2[i, j]:
                rows.append(i); cols.append(j); attrs.append([z[i], z[j], 2.0])

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    edge_attr = torch.tensor(np.asarray(attrs), dtype=torch.float32)
    return edges, edge_attr


def compute_unimol1_node_feat(symbols, coords, device):
    unimol = UniMolRepr(data_type="molecule", model_name="unimolv1", remove_hs=False, use_cuda=torch.cuda.is_available())
    out = unimol.get_repr({"atoms": [symbols], "coordinates": [coords.tolist()]}, return_atomic_reprs=True)

    emb = np.asarray(out["atomic_reprs"][0], dtype=np.float32)
    n = len(symbols)

    if emb.shape[0] == n + 1:
        emb = emb[:-1]      # convention que tu utilises déjà dans dataset.py
    elif emb.shape[0] != n:
        raise ValueError(f"UniMol atomic_reprs shape {emb.shape}, attendu {n} ou {n+1}.")

    z = symbols_to_z(symbols)[:, None]
    feat = np.concatenate([z, emb], axis=1)
    return torch.tensor(feat, dtype=torch.float32, device=device)


def prepare_inputs_from_xyz(symbols, coords, device):
    edges, edge_attr = build_graph_from_coords(symbols, coords)
    node_feat_unimol = compute_unimol1_node_feat(symbols, coords, device)
    return {
        "symbols": symbols,
        "coords": coords,
        "n_nodes": len(symbols),
        "edges": [e.to(device) for e in edges],
        "edge_attr": edge_attr.to(device),
        "node_feat_unimol": node_feat_unimol,
        "node_feat_classic": node_feat_unimol[:, :1],
    }


def prepare_inputs_from_dataset(dataset, device):
    sample = dataset[0]
    if len(sample) == 5:
        _, edge_attr, node_feat, _, _ = sample
    else:
        _, edge_attr, node_feat, _ = sample

    node_feat = node_feat.to(device, dtype=torch.float32).reshape(dataset.n_node, -1)
    edges = dataset.get_edges(1, dataset.n_node)
    return {
        "symbols": get_dataset_symbols(dataset),
        "n_nodes": dataset.n_node,
        "edges": [e.to(device) for e in edges],
        "edge_attr": edge_attr.to(device),
        "node_feat_unimol": node_feat,
        "node_feat_classic": node_feat[:, :1],
    }


# ============================================================
# MODELS
# ============================================================

def build_estag_model(cfg, n_nodes, edge_attr, device, num_past):
    state = load_state(cfg["checkpoint"], device)
    hidden_nf = infer_hidden_nf(state) if cfg["hidden_nf"] is None else cfg["hidden_nf"]
    fft = infer_fft(state, num_past) if cfg["fft"] == "auto" else bool(cfg["fft"])
    in_edge_nf = num_past - 1 if fft else edge_attr.shape[-1]
    nodes_att_dim = num_past - 1 if fft else 0

    model = cfg["class"](num_past=num_past, num_future=1, in_node_nf=cfg["in_node_nf"],
                         in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, fft=fft, eat=True,
                         nodes_att_dim=nodes_att_dim, device=device, n_layers=cfg["n_layers"],
                         n_nodes=n_nodes).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"{cfg['name']} chargé | hidden_nf={hidden_nf} | fft={fft} | in_node_nf={cfg['in_node_nf']}")
    return model


def make_mlatom_mol(symbols, coords, velocities):
    mol = ml.data.molecule.from_numpy(coordinates=np.asarray(coords, dtype=np.float32), species=np.asarray(symbols))
    mol.add_xyz_vectorial_property(np.asarray(velocities, dtype=np.float32), "xyz_velocities")
    return mol


def collapse_mdtrajnet_array(x, n_atoms):
    x = np.asarray(x)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim == 3 and x.shape[0] == n_atoms and x.shape[1] == 3:
        return x.mean(axis=-1).astype(np.float32)
    if x.ndim == 3 and x.shape[1] == n_atoms and x.shape[2] == 3:
        return x.mean(axis=0).astype(np.float32)
    raise ValueError(f"Shape MDTrajNet inattendue : {x.shape}")


# ============================================================
# ROLLOUTS
# ============================================================

@torch.no_grad()
def rollout_estag(model, node_feat, edge_attr, edges, frame0, steps, device):
    frame0_t = torch.tensor(frame0, dtype=torch.float32, device=device)
    current = frame0_t.unsqueeze(0).repeat(ARGS.num_past, 1, 1)
    frames, values = [frame0.copy()], {}

    for step in range(1, max(steps) + 1):
        pred = model(node_feat, current, edges, edge_attr)
        pred_np = pred.detach().cpu().numpy()
        frames.append(pred_np.copy())
        if step in steps:
            values[step] = mse(pred_np, frame0)
        current = torch.cat([current[1:], pred.unsqueeze(0)], dim=0)

    return values, frames


def make_mlatom_mol(symbols, coords, velocities):
    mol = ml.data.molecule.from_numpy(
        coordinates=np.asarray(coords, dtype=np.float32),
        species=np.asarray(symbols),
    )
    mol.add_xyz_vectorial_property(
        np.asarray(velocities, dtype=np.float32),
        "xyz_velocities",
    )
    return mol


def rollout_mdtrajnet(model, symbols, frame0, estag_steps, dt_fs):
    """
    Version conforme à l'usage officiel MDTrajNet-1.

    On donne :
        - coordonnées initiales
        - vitesses initiales nulles

    Puis on laisse MLatom gérer :
        - propagate()
        - segmentation temporelle
        - appel interne à predict()
        - sélection/stabilisation de l'ensemble via ANI-1ccx
    """

    max_step = max(estag_steps)
    max_time = max_step * dt_fs

    velocities0 = np.zeros_like(frame0, dtype=np.float32)
    mol0 = make_mlatom_mol(symbols, frame0, velocities0)

    ani1ccx = ml.methods("ANI-1ccx")

    traj = model.propagate(
        molecule=mol0,
        time=max_time,
        time_step=dt_fs,
        time_segment=2 * dt_fs,
        rescale_velocities=True,
        potential_model=ani1ccx,
        batch_size=0,
    )

    db = traj.to_database()

    values = {}
    frames = [frame0.copy()]

    for step in estag_steps:
        if step >= len(db):
            values[step] = np.nan
            continue

        coords = np.asarray(db[step].xyz_coordinates, dtype=np.float32)
        values[step] = mse(coords, frame0)
        frames.append(coords.copy())

    return values, frames


# ============================================================
# START FRAMES
# ============================================================

def get_start_frames_xyz(path):
    symbols, coords = read_xyz_one_frame(path)
    return symbols, [("provided_xyz", coords)]


def get_start_frames_md17(dataset, n_starts, delta_frame, num_past):
    starts = []
    for i in range(len(dataset)):
        if hasattr(dataset, "st"):
            frame_id = int(dataset.st[i]) + delta_frame * (num_past - 1)
        else:
            frame_id = i
        if frame_id < len(dataset.x):
            starts.append((frame_id, get_frame(dataset, frame_id)))
        if len(starts) >= n_starts:
            break
    return get_dataset_symbols(dataset), starts


# ============================================================
# MAIN
# ============================================================

def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["xyz", "md17"], default="xyz")
    p.add_argument("--xyz", default="resultats/zero_mouvement_compare/aspirin_stable_rdkit.xyz")
    p.add_argument("--data_dir", default="md17")
    p.add_argument("--mol", default="aspirin")
    p.add_argument("--result_dir", default="resultats/zero_mouvement_compare")
    p.add_argument("--num_starts", type=int, default=100)
    p.add_argument("--num_past", type=int, default=10)
    p.add_argument("--delta_frame", type=int, default=10)
    p.add_argument("--md17_dt_fs", type=float, default=0.5)
    p.add_argument("--steps", default="1,11,100,200")
    p.add_argument("--classic_ckpt", default="logs/md17_logs/exp_aspirin/estag/saved_classical_model_weights.pt")
    p.add_argument("--unimol_ckpt", default="resultats/estag_unimol_rollout_fft_aspirine.pt")
    p.add_argument("--mdtrajnet_model", default="CHEMIN/TEMPORAIRE/MDtrajNet1")
    p.add_argument("--save_examples", type=int, default=1)
    ARGS = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [int(x) for x in ARGS.steps.split(",")]
    os.makedirs(ARGS.result_dir, exist_ok=True)

    if ARGS.mode == "xyz":
        symbols, starts = get_start_frames_xyz(ARGS.xyz)
        inputs = prepare_inputs_from_xyz(symbols, starts[0][1], device)
    else:
        dataset = MD17Dataset(partition="test", max_samples=max(ARGS.num_starts, 1000),
                              data_dir=ARGS.data_dir, molecule_type=ARGS.mol,
                              delta_frame=ARGS.delta_frame, num_past=ARGS.num_past)
        symbols, starts = get_start_frames_md17(dataset, ARGS.num_starts, ARGS.delta_frame, ARGS.num_past)
        inputs = prepare_inputs_from_dataset(dataset, device)

    dt_fs = ARGS.delta_frame * ARGS.md17_dt_fs
    print("mode:", ARGS.mode, "| starts:", len(starts), "| dt per model step:", dt_fs, "fs")

    estag_cfgs = [
        {"name":"estag_classic", "checkpoint":ARGS.classic_ckpt, "class":ESTAGClassic,
         "in_node_nf":1, "hidden_nf":None, "fft":"auto", "n_layers":2},
        {"name":"estag_unimol", "checkpoint":ARGS.unimol_ckpt, "class":ESTAGUnimol,
         "in_node_nf":513, "hidden_nf":None, "fft":"auto", "n_layers":2},
    ]

    models = []
    for cfg in estag_cfgs:
        if os.path.isfile(cfg["checkpoint"]):
            m = build_estag_model(cfg, inputs["n_nodes"], inputs["edge_attr"], device, ARGS.num_past)
            feat = inputs["node_feat_classic"] if cfg["in_node_nf"] == 1 else inputs["node_feat_unimol"]
            models.append(("estag", cfg["name"], m, feat))
        else:
            print("[skip]", cfg["name"], "checkpoint introuvable:", cfg["checkpoint"])

    mdtrajnet = None
    try:
        mdtrajnet = ml.MDtrajNet(model_file=ARGS.mdtrajnet_model, verbose=False)
        print("mdtrajnet chargé:", ARGS.mdtrajnet_model)
    except Exception as e:
        print("[skip] mdtrajnet:", e)

    per_rows, summary_rows = [], []

    for kind, name, model, feat in models:
        sums = {s:0.0 for s in steps}
        for j, (start_id, frame0) in enumerate(starts):
            vals, frames = rollout_estag(model, feat, inputs["edge_attr"], inputs["edges"], frame0, steps, device)
            row = {"model":name, "start_id":start_id}
            for s in steps:
                row[f"mse_step_{s}"] = vals[s]
                sums[s] += vals[s]
            per_rows.append(row)
            if j < ARGS.save_examples:
                write_xyz(os.path.join(ARGS.result_dir, f"{name}_example_{j}.xyz"), symbols, frames, name)
            print(f"{name} | {j+1}/{len(starts)}")
        for s in steps:
            summary_rows.append([name, len(starts), s, s * dt_fs, sums[s] / len(starts)])

    if mdtrajnet is not None:
        name = "mdtrajnet"
        vals_by_step = {s: [] for s in steps}
        for j, (start_id, frame0) in enumerate(starts):
            vals, frames = rollout_mdtrajnet(
                    mdtrajnet,
                    symbols,
                    frame0,
                    steps,
                    ARGS.delta_frame * ARGS.md17_dt_fs,
                )
            row = {"model":name, "start_id":start_id}
            for s in steps:
                row[f"mse_step_{s}"] = vals[s]
                vals_by_step[s].append(vals[s])
            per_rows.append(row)
            if j < ARGS.save_examples:
                write_xyz(os.path.join(ARGS.result_dir, f"{name}_example_{j}.xyz"), symbols, frames, name)
            print(f"{name} | {j+1}/{len(starts)}")
        for s in steps:
            arr = np.asarray(vals_by_step[s], dtype=np.float64)
            mean = np.nanmean(arr) if np.any(np.isfinite(arr)) else np.nan
            n_valid = int(np.sum(np.isfinite(arr)))

            summary_rows.append([name, n_valid, s, s * dt_fs, mean])

    per_path = os.path.join(ARGS.result_dir, f"zero_mouvement_{ARGS.mode}_per_start.csv")
    sum_path = os.path.join(ARGS.result_dir, f"zero_mouvement_{ARGS.mode}_summary.csv")

    fields = ["model", "start_id"] + [f"mse_step_{s}" for s in steps]
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_rows)

    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "num_valid_starts", "step", "physical_time_fs", "mean_mse_vs_initial"])
        w.writerows(summary_rows)

    print("\nRésumé sauvegardé:", sum_path)
    for r in summary_rows:
        print(f"{r[0]:15s} step={r[2]:4d} t={r[3]:8.2f} fs mean_mse={r[4]:.8f}")


if __name__ == "__main__":
    main()