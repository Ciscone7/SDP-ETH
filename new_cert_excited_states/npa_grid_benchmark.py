"""
2D NPA grid benchmark: moment matrix ∈ {NPA1, NPA2, NPA3} x shell ∈ {NPA1, NPA2, NPA3, NPA4}.

Produces two figures:
  Figure 1 — 4x4 grid of heatmaps (one per eigenstate). Each heatmap shows
              the certification gap width for every (moment, shell) combination.
  Figure 2 — Single heatmap of total width (summed over all eigenstates).

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.npa_grid_benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.pauli_logic import local_pauli

from new_cert_excited_states.energy_shell_sdp import bound_observable_excited

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N         = 4
J         = 1.0
h         = 0.5
g         = 1.05   # Kim & Huse (2013) chaotic regime
BC        = "open"
DELTA_E   = 0.15
MOSEK_TOL = 1e-8

MOMENT_LEVELS = [1, 2, 3]
SHELL_LEVELS  = [1, 2, 3, 4]

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag      = f"N{N}_g{int(g*100):03d}_{BC}_dE{int(DELTA_E*100):03d}"
out_npz  = OUTPUT_DIR / f"npa_grid_benchmark_{tag}.npz"
out_pdf1 = out_npz.with_name(out_npz.stem + "_per_eigenstate.pdf")
out_pdf2 = out_npz.with_name(out_npz.stem + "_total_width.pdf")

# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

sym = SymmetryManager(N=N, use_real_basis=True)

moment_bases = {lv: generate_npa_basis(N, k=lv).words for lv in MOMENT_LEVELS}
shell_bases  = {lv: generate_npa_basis(N, k=lv).words for lv in SHELL_LEVELS}

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"Moment levels: {MOMENT_LEVELS}  —  sizes: {[len(moment_bases[l]) for l in MOMENT_LEVELS]}")
print(f"Shell  levels: {SHELL_LEVELS}   —  sizes: {[len(shell_bases[l])  for l in SHELL_LEVELS]}")
print(f"Total SDPs: {len(MOMENT_LEVELS) * len(SHELL_LEVELS) * 2**N}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Exact diagonalisation
# ---------------------------------------------------------------------------

H_dict  = xx_tfising_hamiltonian_dict(N, J=J, h=h, g=g, boundary=BC)
H_exact = xx_tfising_hamiltonian_exact(N, J=J, h=h, g=g, boundary=BC)
sz0_op  = qt.tensor([qt.sigmaz()] + [qt.qeye(2)] * (N - 1))

evals_raw, evecs = H_exact.eigenstates()
evals     = np.array([float(e) for e in evals_raw])
obs_exact = np.array([float(qt.expect(sz0_op, v)) for v in evecs])

order     = np.argsort(evals)
evals     = evals[order]
obs_exact = obs_exact[order]

n_states = len(evals)

# ---------------------------------------------------------------------------
# Main computation  — widths[i_moment, i_shell, i_state]
# ---------------------------------------------------------------------------

n_m = len(MOMENT_LEVELS)
n_s = len(SHELL_LEVELS)
widths = np.full((n_m, n_s, n_states), np.nan)

t_total = time.time()

for im, ml in enumerate(MOMENT_LEVELS):
    for is_, sl in enumerate(SHELL_LEVELS):
        t0 = time.time()
        print(f"\nMoment=NPA{ml} ({len(moment_bases[ml])} words)  "
              f"Shell=NPA{sl} ({len(shell_bases[sl])} words)")

        for i, E in enumerate(evals):
            res = bound_observable_excited(
                moment_bases[ml], H_dict, O_dict, E, DELTA_E,
                sym, shell_basis=shell_bases[sl],
                use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
            )
            lb, ub = res.lb, res.ub
            if not (np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub)):
                widths[im, is_, i] = ub - lb

        total_w = np.nansum(widths[im, is_, :])
        print(f"  Total width: {total_w:.4f}   ({time.time()-t0:.1f} s)")

print(f"\nAll done in {time.time()-t_total:.1f} s")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

np.savez(out_npz, widths=widths, evals=evals, obs_exact=obs_exact)
meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
    "moment_levels": MOMENT_LEVELS,
    "shell_levels": SHELL_LEVELS,
    "moment_sizes": {lv: len(moment_bases[lv]) for lv in MOMENT_LEVELS},
    "shell_sizes":  {lv: len(shell_bases[lv])  for lv in SHELL_LEVELS},
    "observable": "Z_0",
    "total_widths": {
        f"m{ml}_s{sl}": float(np.nansum(widths[im, is_, :]))
        for im, ml in enumerate(MOMENT_LEVELS)
        for is_, sl in enumerate(SHELL_LEVELS)
    },
}
out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
print(f"Saved → {out_npz}")

# ---------------------------------------------------------------------------
# Figure 1 — 4x4 grid of per-eigenstate heatmaps
# ---------------------------------------------------------------------------

# Shared color scale across all eigenstates
vmin = 0.0
vmax = np.nanmax(widths)

nrows, ncols = 4, 4   # 16 eigenstates
fig1, axes = plt.subplots(nrows, ncols, figsize=(14, 11))
fig1.suptitle(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}\n"
    f"Certification gap width per eigenstate  —  rows: moment level, cols: shell level",
    fontsize=11,
)

moment_labels = [f"NPA{l}" for l in MOMENT_LEVELS]
shell_labels  = [f"NPA{l}" for l in SHELL_LEVELS]

for idx in range(n_states):
    row, col = divmod(idx, ncols)
    ax = axes[row, col]

    data = widths[:, :, idx]   # shape (n_moment, n_shell)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin, vmax=vmax, origin="upper")

    ax.set_title(
        f"#{idx+1}  E={evals[idx]:+.3f}\n⟨Z₀⟩={obs_exact[idx]:+.3f}",
        fontsize=7, pad=3,
    )
    ax.set_xticks(range(n_s))
    ax.set_yticks(range(n_m))

    if row == nrows - 1:
        ax.set_xticklabels(shell_labels, fontsize=6, rotation=45)
        ax.set_xlabel("shell", fontsize=7)
    else:
        ax.set_xticklabels([])

    if col == 0:
        ax.set_yticklabels(moment_labels, fontsize=6)
        ax.set_ylabel("moment", fontsize=7)
    else:
        ax.set_yticklabels([])

    # Annotate each cell with the width value
    for im_idx, ml in enumerate(MOMENT_LEVELS):
        for is_idx in range(n_s):
            val = data[im_idx, is_idx]
            if not np.isnan(val):
                ax.text(is_idx, im_idx, f"{val:.3f}",
                        ha="center", va="center", fontsize=5,
                        color="black" if val < 0.6 * vmax else "white")

# Shared colorbar
cbar = fig1.colorbar(
    plt.cm.ScalarMappable(cmap="RdYlGn_r",
                          norm=plt.Normalize(vmin=vmin, vmax=vmax)),
    ax=axes, fraction=0.02, pad=0.02,
)
cbar.set_label("gap width (ub − lb)", fontsize=9)

plt.savefig(out_pdf1, bbox_inches="tight")
plt.show()
print(f"Figure 1 saved → {out_pdf1}")

# ---------------------------------------------------------------------------
# Figure 2 — heatmap of total width
# ---------------------------------------------------------------------------

total_width_grid = np.array([
    [np.nansum(widths[im, is_, :]) for is_ in range(n_s)]
    for im in range(n_m)
])

fig2, ax2 = plt.subplots(figsize=(7, 4))
fig2.suptitle(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}\n"
    f"Total certification width (sum over all eigenstates)",
    fontsize=11,
)

im2 = ax2.imshow(total_width_grid, aspect="auto", cmap="RdYlGn_r", origin="upper")
ax2.set_xticks(range(n_s))
ax2.set_yticks(range(n_m))
ax2.set_xticklabels(shell_labels, fontsize=10)
ax2.set_yticklabels(moment_labels, fontsize=10)
ax2.set_xlabel("shell level", fontsize=11)
ax2.set_ylabel("moment level", fontsize=11)

for im_idx in range(n_m):
    for is_idx in range(n_s):
        val = total_width_grid[im_idx, is_idx]
        ax2.text(is_idx, im_idx, f"{val:.3f}",
                 ha="center", va="center", fontsize=10,
                 color="black" if val < 0.6 * total_width_grid.max() else "white")

fig2.colorbar(im2, ax=ax2, label="total gap width")
plt.tight_layout()
plt.savefig(out_pdf2, bbox_inches="tight")
plt.show()
print(f"Figure 2 saved → {out_pdf2}")
