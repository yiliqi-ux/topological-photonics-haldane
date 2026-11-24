"""M2: compute lower and upper band Chern numbers by the FHS algorithm.

Reuse M1's Hamiltonian and Brillouin zone grid in a periodic sublattice basis.
Check integer Chern numbers, cancellation between bands and the transition
near the M1 bulk gap closure.
"""

from pathlib import Path

import numpy as np

from haldane import (
    CRITICAL_MASS, MASS_MAX, MASS_MIN, MASS_SCAN_POINTS, NEAREST, PHI, T1, T2,
    brillouin_zone_grid, hamiltonian,
)
GRID_SIZE = 40


def periodic_hamiltonian(k, mass):
    """Express the M1 Hamiltonian in a reciprocal-periodic sublattice basis."""
    matrix = hamiltonian(k, mass=mass, t1=T1, t2=T2, phi=PHI)
    # Reference each A-to-B hopping to δ1, so δi − δ1 is a Bravais vector.
    matrix[..., 0, 1] *= np.exp(-1j * (k @ NEAREST[0]))
    matrix[..., 1, 0] = matrix[..., 0, 1].conj()
    return matrix


def link_variable(states, axis):
    """Evaluate the normalized forward overlap, wrapping the specified axis."""
    neighbor = np.roll(states, -1, axis=axis)
    overlap = np.sum(states.conj() * neighbor, axis=-1)
    magnitude = np.abs(overlap)
    if np.any(magnitude <= 1e-14):
        raise ValueError("FHS link overlap is zero to numerical precision. Use a finer grid.")
    return overlap / magnitude


def fhs_chern(states):
    """Sum the principal plaquette phases for one band of shape (n, n, 2)."""
    u1 = link_variable(states, axis=0)
    u2 = link_variable(states, axis=1)
    plaquette = u1 * np.roll(u2, -1, axis=0) * np.roll(u1, -1, axis=1).conj() * u2.conj()
    return float(np.angle(plaquette).sum() / (2 * np.pi))


def band_chern_numbers(mass, n):
    """Diagonalize once and evaluate the two band columns independently."""
    if n < 2:
        raise ValueError("The grid must have at least two points per direction.")
    # Band-resolved Chern numbers are undefined at the gap closure, even
    # when the chosen grid does not sample the valley exactly.
    if np.isclose(abs(mass), CRITICAL_MASS, rtol=0, atol=1e-12):
        return np.nan, np.nan

    k = brillouin_zone_grid(n)
    _, eigenvectors = np.linalg.eigh(periodic_hamiltonian(k, mass))
    return fhs_chern(eigenvectors[..., :, 0]), fhs_chern(eigenvectors[..., :, 1])


def chern_number(mass, n=GRID_SIZE, band=0):
    """Return the raw FHS sum for band 0 (lower) or 1 (upper), or NaN at closure."""
    if band not in (0, 1):
        raise ValueError("Select band 0 (lower) or 1 (upper).")
    return band_chern_numbers(mass, n)[band]


def main():
    import matplotlib.pyplot as plt

    masses = np.linspace(MASS_MIN, MASS_MAX, MASS_SCAN_POINTS)
    chern = np.array([band_chern_numbers(mass, GRID_SIZE) for mass in masses])
    print("     M     C_lower    C_upper        Sum")
    for mass in (-1.0, -0.5, 0.0, 0.5, 1.0):
        index = np.argmin(np.abs(masses - mass))
        lower, upper = chern[index]
        # Round only for display to avoid printing numerical noise as -0.
        print(f"{mass:+6.1f} {round(lower):+11d} {round(upper):+10d} {round(lower + upper):+10d}")

    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    # Join the actual mass samples, so each sloped transition spans one scan step.
    axis.plot(masses, chern[:, 0], color="#20579d", lw=2, label="Lower band")
    axis.plot(masses, chern[:, 1], color="#af4e26", lw=2, ls="--", label="Upper band")
    for mass in (-CRITICAL_MASS, CRITICAL_MASS):
        axis.axvline(mass, color="#546273", ls=":", lw=1.5,
                     label="Analytic gap closure" if mass < 0 else None)
        axis.text(mass, 1.18, f"{mass:+.4f}", color="#546273", ha="center", va="bottom",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    axis.set(
        title=f"Haldane band Chern numbers ({GRID_SIZE} × {GRID_SIZE} FHS grid)",
        xlabel=r"Mass $M$ / $t_1$",
        ylabel="Chern number",
        xlim=(masses[0], masses[-1]),
        ylim=(-1.4, 1.42),
        yticks=[-1, 0, 1],
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower left", frameon=False)
    figure.savefig(Path(__file__).with_name("m2_chern.png"), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
