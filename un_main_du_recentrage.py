# main_aspirin_centered_rollout.py

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from md17.dataset import MD17Dataset
from models.model_unimol import ESTAG


# ============================================================
# PARAMÈTRES
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_PAST = 10
DELTA_FRAME = 10
BATCH_SIZE = 32

EPOCHS = 500
LR = 5e-4

DATA_DIR = "md17"
MOLECULE = "aspirin"

MODEL_OUT = "resultats/centre/estag_unimol_rollout_prog_aligne_targetv2.pt"
LOG_FILE = "resultats/centre/best_model_log_aligne_targetv2.txt"

IN_NODE_NF = 513
IN_EDGE_NF = 3
HIDDEN_NF = 32
N_LAYERS = 2


# ============================================================
# DATASETS
# ============================================================

dataset = MD17Dataset(
    partition="train",
    max_samples=42000,
    data_dir=DATA_DIR,
    molecule_type=MOLECULE,
    delta_frame=DELTA_FRAME,
    num_past=NUM_PAST,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataset = MD17Dataset(
    partition="test",
    max_samples=1000,
    data_dir=DATA_DIR,
    molecule_type=MOLECULE,
    delta_frame=DELTA_FRAME,
    num_past=NUM_PAST,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

n_nodes = dataset.n_node


# ============================================================
# MODÈLE
# ============================================================

model = ESTAG(
    num_past=NUM_PAST,
    num_future=1,
    in_node_nf=IN_NODE_NF,
    in_edge_nf=3,
    hidden_nf=HIDDEN_NF,
    fft=False,
    eat=True,
    device=DEVICE,
    n_layers=N_LAYERS,
    n_nodes=n_nodes,
).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
)

criterion = nn.MSELoss()

best_loss = [1e9] * 64


# ============================================================
# ALIGNEMENT KABSCH
# ============================================================

def kabsch_align_torch(mobile, reference):
    """
    Aligne mobile sur reference.

    mobile    : (B, N, 3)
    reference : (B, N, 3)

    Retourne mobile aligné sur reference.
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

    return torch.matmul(X, R) + reference_center


def align_flat_frame_to_reference(frame, reference, batch_size, n_nodes):
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


def center_flat_frame_to_reference(frame, reference, batch_size, n_nodes):
    """
    Variante translation seule.
    Non utilisée par défaut.

    frame     : (B*N, 3)
    reference : (B*N, 3)
    """

    frame_b = frame.view(batch_size, n_nodes, 3)
    ref_b = reference.view(batch_size, n_nodes, 3)

    frame_com = frame_b.mean(dim=1, keepdim=True)
    ref_com = ref_b.mean(dim=1, keepdim=True)

    centered = frame_b - frame_com + ref_com

    return centered.reshape(batch_size * n_nodes, 3)


# Choisis ici :
# "kabsch"      : translation + rotation globale retirées
# "translation" : seulement recentrage du centre géométrique
ALIGN_MODE = "kabsch"


def align_like_prediction(frame, reference, batch_size, n_nodes):
    """
    Applique la même correction à la prédiction et à la target.
    """

    if ALIGN_MODE == "kabsch":
        return align_flat_frame_to_reference(
            frame,
            reference,
            batch_size,
            n_nodes,
        )

    if ALIGN_MODE == "translation":
        return center_flat_frame_to_reference(
            frame,
            reference,
            batch_size,
            n_nodes,
        )

    if ALIGN_MODE == "none":
        return frame

    raise ValueError(f"Unknown ALIGN_MODE: {ALIGN_MODE}")


# ============================================================
# ROLLOUT PROGRESSIF
# ============================================================

def get_rollout_steps(epoch):
    tc = 150

    if epoch < tc + 10:
        return 1
    elif epoch < tc + 20:
        return 2
    elif epoch < tc + 30:
        return 4
    elif epoch < tc + 40:
        return 1
    elif epoch < tc + 60:
        return 6
    elif epoch < tc + 80:
        return 1
    elif epoch < tc + 100:
        return 8
    elif epoch < tc + 120:
        return 1
    elif epoch < tc + 140:
        return 10
    elif epoch < tc + 180:
        return 1
    elif epoch < tc + 240:
        return 12
    elif epoch < tc + 280:
        return 14
    else:
        return 1


def prepare_batch(batch, used_dataset):
    """
    Transforme le batch DataLoader en tenseurs ESTAG.

    Retourne :
        current_window : (NUM_PAST, B*N, 3)
        edge_attr_batch : (B*E, 3)
        node_feat : (B*N, F)
        start_idx : (B,)
        edges
        batch_size
    """

    loc, edge_attr, node_feat, _, start_idx = [
        x.to(DEVICE)
        for x in batch
    ]

    start_idx = start_idx.long()

    batch_size = loc.shape[0]

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

    edges = used_dataset.get_edges(
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


def rollout_loss_on_loader(
    model,
    data_loader,
    used_dataset,
    rollout_steps,
    train=True,
):
    """
    Calcule la loss rollout.

    Point crucial :
    - pred est aligné sur current_window[-1]
    - target_frame est alignée sur current_window[-1]
    Donc le modèle travaille sur une aspirin ground truth centrée/alignée
    de la même manière que sa prédiction.
    """

    if train:
        model.train()
    else:
        model.eval()

    weights = torch.full(
        (rollout_steps,),
        1.0 / rollout_steps,
        device=DEVICE,
    )

    total_loss = 0.0
    n_batches = 0

    for batch in data_loader:

        loc, edge_attr, node_feat, _, start_idx = [
            x.to(DEVICE)
            for x in batch
        ]

        start_idx = start_idx.long()

        last_needed_idx = start_idx + DELTA_FRAME * (
            NUM_PAST + rollout_steps - 1
        )

        valid_mask = last_needed_idx < len(used_dataset.x)

        if not valid_mask.any():
            continue

        loc = loc[valid_mask]
        edge_attr = edge_attr[valid_mask]
        node_feat = node_feat[valid_mask]
        start_idx = start_idx[valid_mask]

        batch_size = loc.shape[0]

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

        edges = used_dataset.get_edges(
            batch_size,
            n_nodes,
        )
        edges = [e.to(DEVICE) for e in edges]

        if train:
            optimizer.zero_grad()

        rollout_loss = 0.0

        for k in range(rollout_steps):

            pred_raw = model(
                node_feat,
                current_window,
                edges,
                edge_attr_batch,
            )

            reference_frame = current_window[-1]

            pred = align_like_prediction(
                pred_raw,
                reference_frame,
                batch_size,
                n_nodes,
            )

            target_indices = (
                start_idx
                + DELTA_FRAME * (NUM_PAST + k)
            )

            target_np = used_dataset.x[
                target_indices.detach().cpu().numpy()
            ]

            target_frame_raw = torch.as_tensor(
                target_np,
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(batch_size * n_nodes, 3)

            target_frame = align_like_prediction(
                target_frame_raw,
                reference_frame,
                batch_size,
                n_nodes,
            )

            rollout_loss = rollout_loss + weights[k] * criterion(
                pred,
                target_frame,
            )

            current_window = torch.cat(
                [
                    current_window[1:],
                    pred.unsqueeze(0),
                ],
                dim=0,
            )

        if train:
            rollout_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        total_loss += rollout_loss.item()
        n_batches += 1

    if n_batches == 0:
        return float("nan")

    return total_loss / n_batches


# ============================================================
# TRAINING
# ============================================================

for epoch in range(1, EPOCHS):

    rollout_steps = get_rollout_steps(epoch)

    train_loss = rollout_loss_on_loader(
        model=model,
        data_loader=loader,
        used_dataset=dataset,
        rollout_steps=rollout_steps,
        train=True,
    )

    with torch.no_grad():
        val_loss = rollout_loss_on_loader(
            model=model,
            data_loader=val_loader,
            used_dataset=val_dataset,
            rollout_steps=rollout_steps,
            train=False,
        )

    print(
        f"epoch {epoch:4d} | "
        f"loss = {train_loss:.8f} | "
        f"val_loss = {val_loss:.8f} | "
        f"rollout = {rollout_steps:2d} | "
        f"align = {ALIGN_MODE}"
    )

    if val_loss < best_loss[rollout_steps]:

        best_loss[rollout_steps] = val_loss

        torch.save(
            model.state_dict(),
            MODEL_OUT,
        )

        with open(LOG_FILE, "a") as record_log:
            record_log.write(
                f"epoch={epoch} "
                f"rollout_steps={rollout_steps} "
                f"train_loss={train_loss:.8e} "
                f"val_loss={val_loss:.8e} "
                f"align_mode={ALIGN_MODE}\n"
            )

        print(
            f"saved model "
            f"(rollout={rollout_steps}, "
            f"loss={best_loss[rollout_steps]:.8f})"
        )

print()
print("training finished")
print("best loss =", best_loss)