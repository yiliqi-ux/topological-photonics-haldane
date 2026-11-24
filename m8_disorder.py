"""M8: compare late upstream probability under quenched box disorder.

Reuse M7's clean ribbon and chain, input packets, propagation, flight
distances and spatial probability windows. Compare 50 seeded realizations
at each disorder strength and show one real-space sample alongside the mean.
"""

from pathlib import Path

import numpy as np
from scipy.sparse import diags

from haldane import T1
from m7_defect import (
    CRITICAL_MASS, DEFECT_CELLS, FLIGHT_CELLS, FRAMES, GUARD_CELLS, LAUNCH_CELL, LENGTH, WIDTH,
    chain_packet, edge_packet, evolve, finite_ribbon, scattering_probabilities,
)


DISORDER_WIDTHS = np.linspace(0, 2, 9) * T1
REALIZATIONS = 50
BASE_SEED = 20261124
SNAPSHOT_WIDTH = 0.75 * T1
SNAPSHOT_INDEX = 0
LABELS = ("Topological edge", "1D chain")
COLORS = ("#2366ab", "#b9552f")


def disordered_hamiltonian(matrix, strength, rng):
    """Draw one independent onsite potential per site and hold it fixed."""
    onsite = rng.uniform(-strength / 2, strength / 2, size=matrix.shape[0])
    return matrix + diags(onsite, format="csr")


def run_ensemble():
    no_defect = np.empty(0, dtype=int)
    ribbon = finite_ribbon(LENGTH, WIDTH, no_defect, potential=0)
    ribbon_packet, ribbon_speed = edge_packet(LENGTH, WIDTH, LAUNCH_CELL)
    chain, initial_chain = chain_packet(LENGTH, LAUNCH_CELL, no_defect)
    systems = ((ribbon, ribbon_packet), (chain, initial_chain))
    stop_times = FLIGHT_CELLS / np.array([ribbon_speed, 2 * T1])
    reflected = np.empty((len(systems), len(DISORDER_WIDTHS), REALIZATIONS))
    snapshots = np.empty((len(systems), LENGTH))
    max_norm_error = 0.0

    print("  W/t1         Edge mean        Edge std       Chain mean       Chain std", flush=True)
    for i, strength in enumerate(DISORDER_WIDTHS):
        for j in range(REALIZATIONS):
            seed = BASE_SEED + 1000 * i + j
            rng = np.random.default_rng(seed)
            snapshot = strength == SNAPSHOT_WIDTH and j == SNAPSHOT_INDEX
            for system, ((matrix, packet), stop) in enumerate(zip(systems, stop_times)):
                hamiltonian = disordered_hamiltonian(matrix, strength, rng)
                density = evolve(hamiltonian, packet, stop, frames=FRAMES if snapshot else 2)
                norm_error = np.max(np.abs(density.sum(axis=1) - 1))
                max_norm_error = max(max_norm_error, norm_error)
                longitudinal = density[-1].reshape(LENGTH, -1).sum(axis=1)
                # Use M7's measurement windows, with no defect in either Hamiltonian.
                upstream, _, _ = scattering_probabilities(longitudinal, DEFECT_CELLS)
                reflected[system, i, j] = upstream
                if snapshot:
                    snapshots[system] = longitudinal
        means = reflected[:, i].mean(axis=1)
        deviations = reflected[:, i].std(axis=1, ddof=1)
        print(f"{strength / T1:6.2f} {means[0]:17.7g} {deviations[0]:15.7g}"
              f" {means[1]:16.7g} {deviations[1]:15.7g}", flush=True)

    if max_norm_error >= 1e-10:
        raise RuntimeError(f"Probability was not conserved: max error {max_norm_error:.3e}.")
    print(f"Max norm error over both ensembles: {max_norm_error:.3e}")
    return reflected, snapshots, stop_times


def plot_ensemble(reflected, snapshots, stop_times):
    import matplotlib.pyplot as plt

    figure, (scan, realspace) = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    means = reflected.mean(axis=2)
    deviations = reflected.std(axis=2, ddof=1)
    floor = means.min() / 5
    grid = DISORDER_WIDTHS / T1
    for mean, deviation, label, color in zip(means, deviations, LABELS, COLORS):
        # A lower bar crossing zero cannot be shown on a log axis.
        lower = np.maximum(mean - deviation, floor)
        scan.errorbar(grid, mean, yerr=[mean - lower, deviation], color=color, marker="o",
                      capsize=3, lw=1.7, label=label)
    gap = 2 * CRITICAL_MASS / T1
    scan.axvline(gap, color="#647180", ls="--", lw=1.25, label=f"Clean bulk gap ({gap:.4f})")
    scan.set(title=f"(a) Box disorder, {REALIZATIONS} realizations per point",
             xlabel=r"Disorder width $W/t_1$", ylabel=r"Mean upstream probability $\langle R\rangle$",
             yscale="log", ylim=(floor, 1.25), xticks=grid)
    scan.legend(loc="lower center", frameon=False)

    realspace.axvspan(0, DEFECT_CELLS[0] - GUARD_CELLS - 0.5, color="#eaf0f6", zorder=0)
    for density, stop, label, color in zip(snapshots, stop_times, LABELS, COLORS):
        upstream, _, _ = scattering_probabilities(density, DEFECT_CELLS)
        realspace.plot(density, color=color, lw=2,
                       label=rf"{label} ($t t_1={stop * T1:.1f}$, $R={upstream:.3f}$)")
    realspace.axvline(LAUNCH_CELL, color="#647180", lw=1.2, ls=":", label="Launch center")
    realspace.set(title=rf"(b) One realization at $W/t_1={SNAPSHOT_WIDTH/T1:g}$",
                  xlabel="Longitudinal cell", ylabel="Late longitudinal probability",
                  xlim=(0, LENGTH - 1), ylim=(0, None))
    realspace.legend(loc="upper center", frameon=False, fontsize=9)

    for axis in (scan, realspace):
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(Path(__file__).with_name("m8_disorder.png"), dpi=180)
    plt.close(figure)


def main():
    reflected, snapshots, stop_times = run_ensemble()
    plot_ensemble(reflected, snapshots, stop_times)


if __name__ == "__main__":
    main()
