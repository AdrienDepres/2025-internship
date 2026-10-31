# plot_xyz_center_drift.py

import argparse
import numpy as np
import matplotlib.pyplot as plt


ATOMIC_MASSES = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
}


def read_xyz_trajectory(filename):
    """
    Lit un fichier XYZ multi-frame.

    Retourne :
        symbols : list[str], longueur N
        coords  : np.ndarray, shape (T, N, 3)
    """

    frames = []
    symbols_ref = None

    with open(filename, "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            line = line.strip()

            if line == "":
                continue

            n_atoms = int(line)

            # ligne commentaire
            f.readline()

            symbols = []
            coords = []

            for _ in range(n_atoms):

                parts = f.readline().split()

                symbols.append(parts[0])

                coords.append(
                    [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ]
                )

            if symbols_ref is None:
                symbols_ref = symbols
            elif symbols != symbols_ref:
                raise ValueError(
                    "Les symboles atomiques changent entre les frames."
                )

            frames.append(coords)

    coords = np.asarray(frames, dtype=np.float64)

    return symbols_ref, coords


def compute_center_of_mass(symbols, coords, use_masses=True):
    """
    coords : (T, N, 3)

    Retourne :
        centers : (T, 3)
    """

    if use_masses:

        masses = []

        for s in symbols:
            if s not in ATOMIC_MASSES:
                raise ValueError(
                    f"Masse atomique inconnue pour l'élément {s}"
                )

            masses.append(ATOMIC_MASSES[s])

        masses = np.asarray(masses, dtype=np.float64)

        centers = np.sum(
            coords * masses[None, :, None],
            axis=1,
        ) / np.sum(masses)

    else:

        # centre géométrique simple
        centers = coords.mean(axis=1)

    return centers


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "xyz_file",
        help="Fichier XYZ multi-frame",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Pas de temps entre deux frames, par exemple 10.0 en fs",
    )

    parser.add_argument(
        "--geometric",
        action="store_true",
        help="Utiliser le centre géométrique au lieu du centre de masse",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Nom du fichier image à sauvegarder, par exemple drift.png",
    )

    args = parser.parse_args()

    symbols, coords = read_xyz_trajectory(args.xyz_file)

    print("Nombre de frames :", coords.shape[0])
    print("Nombre d'atomes  :", coords.shape[1])

    centers = compute_center_of_mass(
        symbols,
        coords,
        use_masses=not args.geometric,
    )

    drift_from_initial = np.linalg.norm(
        centers - centers[0],
        axis=1,
    )

    step_drift = np.linalg.norm(
        centers[1:] - centers[:-1],
        axis=1,
    )

    if args.dt is None:
        x = np.arange(len(drift_from_initial))
        xlabel = "Frame"
    else:
        x = np.arange(len(drift_from_initial)) * args.dt
        xlabel = "Temps"

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        drift_from_initial,
        label=r"$||C(t)-C(0)||$",
    )

    if args.dt is None:
        plt.plot(
            x[1:],
            step_drift,
            "--",
            label=r"$||C(t)-C(t-1)||$",
        )
    else:
        plt.plot(
            x[1:],
            step_drift,
            "--",
            label=r"$||C(t)-C(t-\Delta t)||$",
        )

    plt.xlabel(xlabel)
    plt.ylabel("Déplacement du centre / Å")
    plt.title("Dérive du centre de masse")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if args.out is not None:
        plt.savefig(args.out, dpi=200)
        print("Graphique sauvegardé :", args.out)

    plt.show()

    print()
    print("Drift final       :", drift_from_initial[-1])
    print("Drift maximal     :", drift_from_initial.max())
    print("Drift moyen/pas   :", step_drift.mean())
    print("Drift max/pas     :", step_drift.max())


if __name__ == "__main__":
    main()