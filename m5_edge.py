"""M5: resolve the chiral edge dispersion of a finite zigzag ribbon.

Reuse M1's bonds and projected bulk Hamiltonian, with the M4 photonic
interpretation. Verify two localized, oppositely moving gap crossings in
the topological phase and zero in-gap states in the trivial phase.
"""

from pathlib import Path

import numpy as np

from haldane import NEAREST, NEXT_NEAREST, PHI, T1, T2, hamiltonian
from m4_photonic import KAPPA1


WIDTH = 40
EDGE_ROWS = 4
EDGE_THRESHOLD = 0.5
# Ribbon x points along the zigzag direction (M1's Cartesian y axis).
DIRECT = np.stack((NEXT_NEAREST[0], NEAREST[0] - NEAREST[2]))
RECIPROCAL = 2 * np.pi * np.linalg.inv(DIRECT).T
PERIOD = np.linalg.norm(DIRECT[0])
B_OFFSET = NEAREST[2]
HARMONICS = np.arange(-1, 2)


def cell_shift(displacement):
    """Express a bond minus its sublattice offset as a Bravais translation."""
    fractional = np.linalg.solve(DIRECT.T, displacement)
    shift = np.rint(fractional).astype(int)
    np.testing.assert_allclose(fractional, shift, atol=1e-12)
    return shift


def hopping_terms(mass):
    """Yield (source, destination, longitudinal shift, row shift, H element)."""
    for bond in NEAREST:
        longitudinal, row = cell_shift(bond - B_OFFSET)
        yield 0, 1, longitudinal, row, T1
    for bond in NEXT_NEAREST:
        longitudinal, row = cell_shift(bond)
        for sublattice, sign in enumerate((1, -1)):
            # For A, the +b bond gives 2*t2*cos(k·b + phi), as in M1.
            yield sublattice, sublattice, longitudinal, row, T2 * np.exp(1j * sign * PHI)
    for sublattice, sign in enumerate((1, -1)):
        # The assembly adds the Hermitian partner, so enter half the onsite term.
        yield sublattice, sublattice, 0, 0, sign * mass / 2


def ribbon_blocks(width, mass):
    """Return the Fourier coefficients for longitudinal shifts −1, 0, +1."""
    blocks = np.zeros((3, 2 * width, 2 * width), dtype=complex)
    for sub_a, sub_b, longitudinal, row, amplitude in hopping_terms(mass):
        rows = np.arange(max(0, -row), min(width, width - row))
        source = 2 * rows + sub_a
        target = 2 * (rows + row) + sub_b
        blocks[1 + longitudinal, source, target] += amplitude
        blocks[1 - longitudinal, target, source] += amplitude.conjugate()
    return blocks


def check_bulk_reconstruction(mass):
    """Restore the transverse period and compare every matrix element to M1."""
    k = np.array([[0.19, -0.61], [0.81, 1.12], [-1.31, 0.51]])
    coordinates = k @ DIRECT.T
    reconstructed = np.zeros((len(k), 2, 2), dtype=complex)
    for a, b, longitudinal, row, amplitude in hopping_terms(mass):
        value = amplitude * np.exp(1j * (coordinates @ [longitudinal, row]))
        reconstructed[:, a, b] += value
        reconstructed[:, b, a] += value.conj()
    expected = hamiltonian(k, mass=mass, t1=T1, t2=T2, phi=PHI)
    expected[:, 0, 1] *= np.exp(-1j * (k @ B_OFFSET))
    expected[:, 1, 0] = expected[:, 0, 1].conj()
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12, rtol=0)
    return np.max(np.abs(reconstructed - expected))


def projected_bulk(q, mass):
    transverse = np.arange(360) / 360
    k = q[:, None, None] / (2 * np.pi) * RECIPROCAL[0] + transverse[:, None] * RECIPROCAL[1]
    bands = np.linalg.eigvalsh(hamiltonian(k, mass=mass, t1=T1, t2=T2, phi=PHI))
    return bands.min(axis=1), bands.max(axis=1)


def localize_edge_pair(pair):
    """Resolve a degenerate pair using the first and last EDGE_ROWS rows."""
    edge_sites = 2 * EDGE_ROWS
    left = pair[:edge_sites]
    right = pair[-edge_sites:]
    _, rotation = np.linalg.eigh(left.conj().T @ left - right.conj().T @ right)
    return pair @ rotation


def ribbon_spectrum(q, mass, gap):
    blocks = ribbon_blocks(WIDTH, mass)
    edge_sites = 2 * EDGE_ROWS
    frequencies = np.empty((len(q), 2 * WIDTH))
    weights = np.empty((*frequencies.shape, 2))
    velocities = np.empty_like(frequencies)
    for index, momentum in enumerate(q):
        phase = np.exp(1j * momentum * HARMONICS)
        matrix = np.einsum("r,rij->ij", phase, blocks)
        derivative = np.einsum("r,rij->ij", 1j * PERIOD * HARMONICS * phase, blocks)
        energy, vectors = np.linalg.eigh(matrix)
        inside = np.flatnonzero((gap[0] < energy) & (energy < gap[1]))
        if len(inside) == 2 and abs(np.diff(energy[inside])[0]) < 1e-10:
            # At an exact (within tolerance) crossing, eigh may mix the two edges.
            # Diagonalize left-minus-right weight within that degenerate eigenspace.
            vectors[:, inside] = localize_edge_pair(vectors[:, inside])
        density = np.abs(vectors) ** 2
        frequencies[index] = energy
        weights[index, :, 0] = density[:edge_sites].sum(axis=0)
        weights[index, :, 1] = density[-edge_sites:].sum(axis=0)
        velocities[index] = np.sum(vectors.conj() * (derivative @ vectors), axis=0).real
    return frequencies, weights, velocities


def edge_branches(frequencies, weights, velocities, gap):
    """Extract the two edge-resolved spectra using total end-row probability."""
    in_gap = (gap[0] < frequencies) & (frequencies < gap[1])
    branches = []
    for boundary in range(2):
        mask = in_gap & (weights[:, :, boundary] > EDGE_THRESHOLD)
        if np.any(mask.sum(axis=1) > 1):
            raise RuntimeError("Multiple in-gap states on the same edge at one momentum.")
        rows, bands = np.nonzero(mask)
        branches.append((rows, frequencies[rows, bands], weights[rows, bands], velocities[rows, bands]))
    return branches


def projected_boundary_error(frequencies, weights, bulk_min, bulk_max):
    """Measure the distance from each projected boundary to a ribbon bulk mode."""
    interior = weights.max(axis=2) <= EDGE_THRESHOLD
    boundaries = np.concatenate((bulk_min, bulk_max), axis=1)
    distance = np.abs(boundaries[:, :, None] - frequencies[:, None, :])
    return np.max(np.min(np.where(interior[:, None, :], distance, np.inf), axis=2))


def main():
    import matplotlib.pyplot as plt

    # Include both projected valleys (q = 2π/3, 4π/3) and the midgap crossing.
    q = np.linspace(0, 2 * np.pi, 601)
    kx = q / PERIOD
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, layout="constrained")
    edge_colors = ["#2065a8", "#b84c27"]
    edge_labels = ["Left edge", "Right edge"]
    for axis, mass, title in zip(axes, (0, 1), (r"(a) Topological, $\Delta/\kappa_1=0$",
                                                r"(b) Trivial, $\Delta/\kappa_1=1$")):
        bulk_min, bulk_max = projected_bulk(q, mass)
        gap = bulk_max[:, 0].max(), bulk_min[:, 1].min()
        frequencies, weights, velocities = ribbon_spectrum(q, mass, gap)
        branches = edge_branches(frequencies, weights, velocities, gap)
        reconstruction_error = check_bulk_reconstruction(mass)
        finite_width_error = projected_boundary_error(frequencies, weights, bulk_min, bulk_max)
        print(f"Delta/kappa1={mass/KAPPA1:g}: projected bulk gap "
              f"[{gap[0]/KAPPA1:+.6f}, {gap[1]/KAPPA1:+.6f}] kappa1")
        print(f"  Bulk reconstruction max error: {reconstruction_error/KAPPA1:.2e} kappa1")
        print(f"  Max projected boundary distance to interior-like ribbon states: {finite_width_error/KAPPA1:.4f} kappa1")
        print(f"  In-gap edge branches: {sum(len(rows) > 0 for rows, *_ in branches)}")

        for band in (0, 1):
            axis.fill_between(kx, bulk_min[:, band] / KAPPA1, bulk_max[:, band] / KAPPA1, color="#dce6f1",
                              label="Projected 2D bulk" if band == 0 else None)
        axis.axhspan(gap[0] / KAPPA1, gap[1] / KAPPA1, color="#faf3dc", alpha=0.65, zorder=0, label="Global bulk gap")
        axis.plot(kx, frequencies / KAPPA1, color="#718094", lw=0.45, alpha=0.42)
        for (rows, energy, localization, velocity), color, label in zip(branches, edge_colors, edge_labels):
            if not len(rows):
                continue
            assert np.all(np.diff(rows) == 1), "Disconnected in-gap branch"
            assert energy.min() < gap[0] + 0.05 * (gap[1] - gap[0])
            assert energy.max() > gap[1] - 0.05 * (gap[1] - gap[0])
            center = np.argmin(np.abs(energy - np.mean(gap)))
            direction = 1 if velocity[center] > 0 else -1
            assert np.all(direction * velocity > 0), "Group velocity changes sign within the gap"
            dl, dr = localization[center]
            minimum_weight = localization.max(axis=1).min()
            print(f"  {label}: span [{energy.min()/KAPPA1:+.4f}, {energy.max()/KAPPA1:+.4f}] kappa1, "
                  f"v_x={velocity[center]/KAPPA1:+.4f} kappa1*a, midgap weights (left, right)=({dl:.6f}, {dr:.6f}), "
                  f"min. own-edge weight={minimum_weight:.6f}")
            axis.plot(kx[rows], energy / KAPPA1, color=color, lw=2.8,
                      label=label + (r" ($+x$)" if direction > 0 else r" ($-x$)"), zorder=4)
            arrow = np.argmin(np.abs(energy - 0.45 * gap[1]))
            start, end = arrow - 11 * direction, arrow + 11 * direction
            axis.annotate("", xy=(kx[rows[end]], energy[end] / KAPPA1), xytext=(kx[rows[start]], energy[start] / KAPPA1),
                          arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16), zorder=5)
        assert sum(len(rows) > 0 for rows, *_ in branches) == (2 if mass == 0 else 0)
        if mass == 0:
            assert branches[0][3][0] * branches[1][3][0] < 0
        axis.set(title=title, xlabel=r"Ribbon momentum $k_x$", xlim=(kx[0], kx[-1]))
        axis.set_xticks(np.pi / PERIOD * np.array([0, 0.5, 1, 1.5, 2]),
                        ["0", r"$\pi/2L$", r"$\pi/L$", r"$3\pi/2L$", r"$2\pi/L$"])
        axis.set_ylim(-3.4, 3.4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncols=2)
    axes[0].set_ylabel(r"Relative frequency $\omega/\kappa_1$")
    figure.suptitle(rf"Zigzag Haldane ribbon  |  $W={WIDTH}$, $L=\sqrt{{3}}\,a$, $\kappa_2/\kappa_1={T2/KAPPA1:g}$, $\phi={PHI / np.pi:g}\pi$",
                    fontsize=14, fontweight="bold")
    figure.savefig(Path(__file__).with_name("m5_edge.png"), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
