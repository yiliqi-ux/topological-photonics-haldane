"""M4: calculate the bands and Chern numbers of a photonic Haldane lattice.

Reuse M1's bulk dispersion and M2's FHS calculation. The photonic phase
diagram must agree with M2 under the parameter mapping below.

Electronic quantity       | Photonic coupled-mode quantity
--------------------------|-----------------------------------------------
t₁ (nearest-neighbor hop)  | κ₁, coupling between neighboring cavities
t₂ exp(iφ)                | κ₂ exp(iφ), complex next-nearest-neighbor coupling
M (onsite mass)           | Δ, with cavity frequencies ω₀ + Δ and ω₀ − Δ
E (eigenenergy)           | ω = ω_lab − ω₀, plotted as ω/κ₁

Magneto-optic bias breaks time reversal. Engineered ring-resonator links
can instead supply synthetic phases in a selected circulation sector.
Use κ₁ = 1, κ₂ = 0.15, φ = π/2 and sweep Δ/κ₁ from −1.5 to 1.5.
This is a lossless tight-binding model for the modes near ω₀.
"""

from pathlib import Path

import numpy as np

from haldane import (
    CRITICAL_MASS, GAP_GRID_SIZE, K, MASS_MAX, MASS_MIN, MASS_SCAN_POINTS, PHI, T1, T2,
    brillouin_zone_grid, hamiltonian, minimum_direct_gap, symmetry_path,
)
from m2_chern import GRID_SIZE, band_chern_numbers


# Use the same dimensionless parameters and reciprocal orientation as M2.
KAPPA1 = T1
KAPPA2 = T2


def main():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(12, 5.1), layout="constrained")
    panels = figure.add_gridspec(1, 2, width_ratios=[1.3, 1])
    band_panels = panels[0].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.06)
    bands_axis = figure.add_subplot(band_panels[0])
    valley_axis = figure.add_subplot(band_panels[1], sharey=bands_axis)
    chern_axis = figure.add_subplot(panels[1])

    path, distance, ticks = symmetry_path()
    offsets = np.linspace(-0.28, 0.28, 81)
    valley_path = -K + np.column_stack((offsets, np.zeros_like(offsets)))
    momenta = brillouin_zone_grid(GAP_GRID_SIZE)
    couplings = dict(t1=KAPPA1, t2=KAPPA2, phi=PHI)

    cases = [
        (0, "#20579d", "-", r"$\Delta=0$  Topological"),
        (-1, "#a74b25", "--", r"$\Delta=-1$  Trivial"),
        (1, "#29736b", "-.", r"$\Delta=+1$  Trivial"),
    ]
    for detuning, color, style, label in cases:
        path_frequencies = np.linalg.eigvalsh(hamiltonian(path, mass=detuning, **couplings))
        valley_frequencies = np.linalg.eigvalsh(hamiltonian(valley_path, mass=detuning, **couplings))
        gap = minimum_direct_gap(detuning, momenta, **couplings) / KAPPA1
        handles = bands_axis.plot(distance, path_frequencies / KAPPA1, color=color, ls=style, lw=1.8)
        handles[0].set_label(f"{label}   |   min. gap {gap:.3f}")
        valley_axis.plot(offsets, valley_frequencies / KAPPA1, color=color, ls=style, lw=1.8)

    bands_axis.set(title="(a) Photonic bands", ylabel=r"Relative frequency $\omega/\kappa_1$",
                   xlim=(distance[0], distance[-1]), ylim=(-3.5, 3.5))
    bands_axis.set_xticks(ticks, [r"$\Gamma$", "$K$", "$M$", r"$\Gamma$"])
    bands_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=9)
    valley_axis.set(title=r"Near $K'$", xlim=(offsets[0], offsets[-1]))
    valley_axis.set_xticks([0], ["$K'$"])
    valley_axis.tick_params(axis="y", left=False, labelleft=False)
    bands_axis.grid(alpha=0.24)
    valley_axis.grid(alpha=0.24)

    detunings = np.linspace(MASS_MIN, MASS_MAX, MASS_SCAN_POINTS)
    chern = np.array([band_chern_numbers(delta, GRID_SIZE) for delta in detunings])
    chern_axis.plot(detunings, chern[:, 0], color="#20579d", lw=2, label="Lower band")
    chern_axis.plot(detunings, chern[:, 1], color="#a74b25", lw=2, ls="--", label="Upper band")
    for delta in (-CRITICAL_MASS, CRITICAL_MASS):
        chern_axis.axvline(delta, color="#697585", lw=1.4, ls="--")
        chern_axis.text(delta, 1.15, f"{delta:+.4f}", ha="center", color="#526071",
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))
    chern_axis.set(title="(b) Photonic Chern numbers", xlabel=r"Cavity detuning $\Delta/\kappa_1$",
                   ylabel="Chern number", xlim=(MASS_MIN, MASS_MAX), ylim=(-1.35, 1.38), yticks=[-1, 0, 1])
    chern_axis.grid(axis="y", alpha=0.24)
    chern_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=2, frameon=False)
    figure.suptitle(rf"Photonic Haldane lattice  |  $\kappa_1={KAPPA1:g}$, $\kappa_2={KAPPA2:g}$, $\phi={PHI / np.pi:g}\pi$",
                    fontsize=15, fontweight="bold")
    figure.savefig(Path(__file__).with_name("m4_photonic.png"), dpi=180)
    plt.close(figure)

    print("Delta/kappa1    C_lower    C_upper")
    for delta in (-1, -0.5, 0, 0.5, 1):
        index = np.argmin(np.abs(detunings - delta))
        lower, upper = chern[index]
        print(f"{delta:+12.1f} {round(lower):+10d} {round(upper):+10d}")


if __name__ == "__main__":
    main()
