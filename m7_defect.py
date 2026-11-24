"""M7: measure wavepacket reflection and transmission past a local defect.

Reuse M5's Haldane hopping terms and right-edge Bloch eigenstates. Propagate
on a finite open ribbon and a matched 1D chain, and check that a strong
onsite barrier reflects the chain packet and transmits the chiral packet.
"""

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply

from haldane import CRITICAL_MASS, T1
from m5_edge import EDGE_ROWS, HARMONICS, PERIOD, hopping_terms, localize_edge_pair, ribbon_blocks


LENGTH = 64
WIDTH = 20
SIGMA = 6.0
POTENTIAL = 3 * T1
DEFECT_CELLS = np.arange(LENGTH // 2 - 1, LENGTH // 2 + 2)
LAUNCH_CELL = LENGTH // 2 - 16
GUARD_CELLS = 3
FLIGHT_CELLS = 33
FRAMES = 2 * FLIGHT_CELLS + 1


def finite_ribbon(length, width, defect_cells, potential):
    """Apply the M5 bonds with open boundaries in both directions (CSR)."""
    rows, columns, values = [], [], []
    for source, target, dx, dy, amplitude in hopping_terms(0):
        x, y = np.mgrid[max(0, -dx):min(length, length - dx),
                        max(0, -dy):min(width, width - dy)]
        i = 2 * (x * width + y).ravel() + source
        j = 2 * ((x + dx) * width + y + dy).ravel() + target
        rows.extend((i, j))
        columns.extend((j, i))
        values.extend((np.full(i.size, amplitude), np.full(i.size, np.conj(amplitude))))

    # A is the outermost sublattice on the high-row (right) zigzag edge.
    defect_sites = 2 * (defect_cells * width + width - 1)
    rows.append(defect_sites)
    columns.append(defect_sites)
    values.append(np.full(defect_sites.size, potential))
    shape = (2 * length * width,) * 2
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                        shape=shape).tocsr()
    matrix.eliminate_zeros()
    return matrix


def edge_packet(length, width, launch):
    """Build a Gaussian spectral packet using only positive-velocity edge modes."""
    blocks = ribbon_blocks(width, mass=0)
    shifts = np.linspace(-5 / SIGMA, 5 / SIGMA, 129)
    momenta = np.pi + shifts
    states = []
    velocities = []
    for q in momenta:
        phases = np.exp(1j * q * HARMONICS)
        hamiltonian = np.einsum("r,rij->ij", phases, blocks)
        energy, vectors = np.linalg.eigh(hamiltonian)
        candidates = np.flatnonzero(np.abs(energy) < CRITICAL_MASS)
        if len(candidates) != 2:
            raise RuntimeError("Launch spectrum extends outside the two gap branches.")
        pair = vectors[:, candidates]
        if np.ptp(energy[candidates]) < 1e-10:
            pair = localize_edge_pair(pair)
        right_weights = np.sum(np.abs(pair[-2 * EDGE_ROWS:]) ** 2, axis=0)
        right = pair[:, right_weights.argmax()]
        # Align all Bloch phases on the outermost A site before superposition.
        right = right * np.exp(-1j * np.angle(right[-2]))
        derivative = np.einsum("r,rij->ij", 1j * HARMONICS * phases, blocks)
        speed = np.vdot(right, derivative @ right).real
        if speed <= 0 or right_weights.max() < 0.5:
            raise RuntimeError("Selected launch mode is not a right-moving right-edge state.")
        states.append(right)
        velocities.append(speed)

    envelope = np.exp(-0.5 * (SIGMA * shifts) ** 2)
    plane_waves = np.exp(1j * np.outer(np.arange(length) - launch, momenta))
    packet = ((plane_waves * envelope) @ np.asarray(states)).ravel()
    return packet / np.linalg.norm(packet), velocities[len(velocities) // 2]


def chain_packet(length, launch, defect_cells):
    """Build the equal-length chain and mid-band right-moving Gaussian packet."""
    diagonal = np.zeros(length)
    diagonal[defect_cells] = POTENTIAL
    hopping = np.full(length - 1, T1)
    matrix = diags([hopping, diagonal, hopping], [-1, 0, 1], format="csr")
    x = np.arange(length) - launch
    # E(q) = 2*t1*cos(q) for positive hopping, so q = −π/2 moves right.
    packet = np.exp(-0.5 * (x / SIGMA) ** 2 - 0.5j * np.pi * x)
    return matrix, packet / np.linalg.norm(packet)


def evolve(matrix, packet, stop, *, frames=FRAMES):
    generator = -1j * matrix
    states = expm_multiply(generator, packet, start=0, stop=stop, num=frames,
                           traceA=generator.diagonal().sum())
    return np.abs(states) ** 2


def scattering_probabilities(density, defect_cells):
    """Split late longitudinal probability outside a buffer around the defect."""
    first = defect_cells[0] - GUARD_CELLS
    after = defect_cells[-1] + GUARD_CELLS + 1
    return np.array([density[:first].sum(), density[after:].sum(), density[first:after].sum()])


def run_experiment(length=LENGTH, launch=LAUNCH_CELL, defect_cells=DEFECT_CELLS):
    matrix = finite_ribbon(length, WIDTH, defect_cells, POTENTIAL)
    packet, speed = edge_packet(length, WIDTH, launch)
    ribbon_stop = FLIGHT_CELLS / speed
    ribbon = evolve(matrix, packet, ribbon_stop).reshape(FRAMES, length, WIDTH, 2).sum(axis=-1)
    chain_matrix, initial_chain = chain_packet(length, launch, defect_cells)
    chain_stop = FLIGHT_CELLS / (2 * T1)
    chain = evolve(chain_matrix, initial_chain, chain_stop)
    return ribbon, chain, ribbon_stop, chain_stop


def plot_experiment(ribbon, chain, ribbon_stop, chain_stop):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    figure = plt.figure(figsize=(12, 6.3), layout="constrained")
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 0.9])
    times = np.linspace(0, ribbon_stop, FRAMES)
    frames = (0, 2 * (LENGTH // 2 - LAUNCH_CELL), FRAMES - 1)
    labels = ("Launch", "At the defect", "After the defect")
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=ribbon[list(frames)].max())
    outer_rows = 8
    axes = []
    for column, (frame, label) in enumerate(zip(frames, labels)):
        axis = figure.add_subplot(grid[0, column])
        axes.append(axis)
        image = axis.imshow(ribbon[frame, :, -outer_rows:].T, origin="lower", aspect="auto",
                            cmap="magma", norm=norm, interpolation="nearest",
                            extent=(-0.5, LENGTH - 0.5, WIDTH - outer_rows - 0.5, WIDTH - 0.5))
        axis.scatter(DEFECT_CELLS, np.full(DEFECT_CELLS.size, WIDTH - 1), marker="s", s=35,
                     facecolors="none", edgecolors="#64eee0", linewidths=1.25)
        axis.set(title=rf"{label}  ($t t_1={times[frame] * T1:.1f}$)", xlabel="Longitudinal cell",
                 yticks=[WIDTH - outer_rows, WIDTH - 4, WIDTH - 1])
        if column == 0:
            axis.set_ylabel("Rightmost eight rows")
        else:
            axis.tick_params(labelleft=False)
    figure.colorbar(image, ax=axes, label=r"Row probability $|\psi_A|^2 + |\psi_B|^2$", shrink=0.87)

    axis = figure.add_subplot(grid[1, :])
    for frame, color, style, label in zip(frames, ("#58667b", "#158376", "#a84225"),
                                         ("--", ":", "-"), labels):
        tick = chain_stop * frame * T1 / (FRAMES - 1)
        axis.plot(chain[frame], color=color, ls=style, lw=1.8, label=rf"{label} ($t t_1={tick:.1f}$)")
    axis.axvspan(DEFECT_CELLS[0] - 0.5, DEFECT_CELLS[-1] + 0.5, color="#d7aa46", alpha=0.24,
                label=r"Three sites at $V=3t_1$")
    axis.set(title="1D chain control", xlabel="Longitudinal site", ylabel=r"Probability $|\psi|^2$",
             xlim=(-0.5, LENGTH - 0.5))
    axis.legend(loc="upper right", ncols=2, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Chiral edge transmission past a local barrier", fontweight="bold", fontsize=15)
    figure.savefig(Path(__file__).with_name("m7_defect.png"), dpi=180)
    plt.close(figure)


def main():
    ribbon, chain, ribbon_stop, chain_stop = run_experiment()
    results = []
    for name, density, stop in (("Topological ribbon", ribbon.sum(axis=-1), ribbon_stop),
                                ("1D chain", chain, chain_stop)):
        reflected, transmitted, near = scattering_probabilities(density[-1], DEFECT_CELLS)
        error = np.max(np.abs(density.sum(axis=1) - 1))
        ends = np.max(density[:, :2].sum(axis=1) + density[:, -2:].sum(axis=1))
        if error > 1e-10:
            raise RuntimeError(f"{name}: propagation did not conserve total probability.")
        print(f"{name} (t*t1={stop * T1:.3f}): R={reflected:.6f}, T={transmitted:.6f}, "
              f"R+T={reflected + transmitted:.6f}, P_near={near:.6f}, "
              f"R+T+P_near={reflected + transmitted + near:.9f}")
        print(f"  Max norm error={error:.2e}, max probability in end cells={ends:.3e}")
        results.append((reflected, transmitted))
    if results[0][0] >= 0.01 or results[0][1] <= 0.98 or results[1][0] <= 0.3:
        raise RuntimeError("The late packets did not meet the scattering criteria.")
    print(f"Launch group velocity = {FLIGHT_CELLS / ribbon_stop * PERIOD / T1:.4f} t1*a")
    plot_experiment(ribbon, chain, ribbon_stop, chain_stop)


if __name__ == "__main__":
    main()
