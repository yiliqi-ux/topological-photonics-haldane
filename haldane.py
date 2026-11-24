"""M1: solve electronic Haldane bands and the bulk direct gap.

Define the lattice and shared model parameters used by M2–M5. Inspect the
Gamma–K–M–Gamma dispersion and verify gap closure at the two critical masses.
"""

from pathlib import Path

import numpy as np


SQRT3 = np.sqrt(3)
NEAREST = np.array([[1, 0], [-0.5, SQRT3 / 2], [-0.5, -SQRT3 / 2]])
NEXT_NEAREST = NEAREST[[1, 2, 0]] - NEAREST[[2, 0, 1]]
RECIPROCAL = 2 * np.pi * np.array([[1 / 3, 1 / SQRT3], [1 / 3, -1 / SQRT3]])
K = (RECIPROCAL[0] - RECIPROCAL[1]) / 3
M_POINT = RECIPROCAL[0] / 2
T1 = 1.0
T2 = 0.15
PHI = np.pi / 2
GAP_GRID_SIZE = 90
MASS_MIN, MASS_MAX = -1.5, 1.5
MASS_SCAN_POINTS = 151
CRITICAL_MASS = abs(2 * T2 * np.sin(PHI) * np.sin(K @ NEXT_NEAREST.T).sum())


def hamiltonian(k, *, mass, t1, t2, phi):
    """Return H(k) for momenta shaped (..., 2), with bond length a = 1."""
    k = np.asarray(k)
    nearest_phase = k @ NEAREST.T
    next_phase = k @ NEXT_NEAREST.T
    d0 = 2 * t2 * np.cos(phi) * np.cos(next_phase).sum(axis=-1)
    d3 = mass - 2 * t2 * np.sin(phi) * np.sin(next_phase).sum(axis=-1)
    # The project convention has H_AB = d1 - i*d2.
    hopping = t1 * np.exp(1j * nearest_phase).sum(axis=-1)

    matrix = np.empty(k.shape[:-1] + (2, 2), dtype=complex)
    matrix[..., 0, 0] = d0 + d3
    matrix[..., 1, 1] = d0 - d3
    matrix[..., 0, 1] = hopping
    matrix[..., 1, 0] = hopping.conj()
    return matrix


def symmetry_path():
    """Sample Gamma–K–M–Gamma and return distance along the path."""
    gamma = np.zeros(2)
    points = [gamma, K, M_POINT, gamma]
    segments = [np.linspace(a, b, 100, endpoint=False) for a, b in zip(points, points[1:])]
    path = np.vstack([*segments, gamma])
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(steps)]
    return path, distance, distance[::100]


def minimum_direct_gap(mass, momenta, **hopping):
    energies = np.linalg.eigvalsh(hamiltonian(momenta, mass=mass, **hopping))
    return np.min(energies[..., 1] - energies[..., 0])


def brillouin_zone_grid(size):
    """Return a reciprocal-cell momentum grid with the endpoint omitted."""
    fractional = np.arange(size) / size
    mesh = np.stack(np.meshgrid(fractional, fractional, indexing="ij"), axis=-1)
    return mesh @ RECIPROCAL


def main():
    import matplotlib.pyplot as plt

    hopping = dict(t1=T1, t2=T2, phi=PHI)
    path, distance, ticks = symmetry_path()

    # A grid divisible by three includes both valleys modulo reciprocal vectors.
    momenta = brillouin_zone_grid(GAP_GRID_SIZE)
    masses = np.linspace(MASS_MIN, MASS_MAX, MASS_SCAN_POINTS)
    gaps = [minimum_direct_gap(mass, momenta, **hopping) for mass in masses]

    figure, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    # With this path and hopping phase, the negative critical mass closes at K.
    band_masses = [0, -CRITICAL_MASS, -1.4 * CRITICAL_MASS]
    for axis, mass in zip(axes.flat, band_masses):
        bands = np.linalg.eigvalsh(hamiltonian(path, mass=mass, **hopping))
        axis.plot(distance, bands[:, 0], color="#20579d", lw=1.7)
        axis.plot(distance, bands[:, 1], color="#af4e26", lw=1.7)
        axis.set(title=f"Mass $M = {mass:.3f}$", ylabel="Energy / $t_1$", ylim=(-3.4, 3.4))
        axis.set_xlim(distance[0], distance[-1])
        axis.set_xticks(ticks, [r"$\Gamma$", "$K$", "$M$", r"$\Gamma$"])
        axis.grid(alpha=0.22)

    gap_axis = axes[1, 1]
    gap_axis.plot(masses, gaps, color="#20579d", lw=1.7)
    for mass in [-CRITICAL_MASS, CRITICAL_MASS]:
        gap = minimum_direct_gap(mass, momenta, **hopping)
        gap_axis.axvline(mass, color="#af4e26", ls="--", lw=1)
        gap_axis.scatter(mass, gap, color="#af4e26", zorder=3)
        print(f"M = {mass:+.9f}, minimum direct gap = {gap:.3e}")
    gap_axis.set(title=f"Minimum direct gap ({GAP_GRID_SIZE} × {GAP_GRID_SIZE} grid)", xlabel="Mass $M$ / $t_1$",
                 ylabel=r"$\min_{\mathbf{k}}(E_+ - E_-)$ / $t_1$")
    gap_axis.grid(alpha=0.22)
    figure.suptitle(rf"Electronic Haldane model ($t_1={T1:g}$, $t_2={T2:g}$, $\phi={PHI / np.pi:g}\pi$)")
    output = Path(__file__).with_name("haldane_m1.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output.name}.")


if __name__ == "__main__":
    main()
