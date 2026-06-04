"""
Diagnostic: shell basis vs moment matrix — which constraint drives the bound?

Two configurations with the same words as NPA2, placed differently:

  Config A — moment matrix = NPA1  |  shell = NPA2
  Config B — moment matrix = NPA2  |  shell = NPA1   (standard NPA2 baseline)

If Config A gives tighter bounds, the shell constraints are the bottleneck
and future optimisation effort should target the shell basis.
If Config B is tighter, the moment matrix is what matters.

Results are printed, saved to _outputs/shell_vs_moment_<tag>.npz, and
a PDF is written next to the npz.

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.shell_vs_moment_diagnostic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

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
g         = 0.3
BC        = "periodic"
DELTA_E   = 0.15
MOSEK_TOL = 1e-8

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag     = f"N{N}_dE{int(DELTA_E * 100):03d}"
out_npz = OUTPUT_DIR / f"shell_vs_moment_{tag}.npz"
out_pdf = out_npz.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

sym        = SymmetryManager(N=N, use_real_basis=True)
basis_npa1 = generate_npa_basis(N, k=1).words
basis_npa2 = generate_npa_basis(N, k=2).words

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"NPA1: {len(basis_npa1)} words  |  NPA2: {len(basis_npa2)} words")
print(
    f"\nConfig A — moment matrix = NPA1 ({len(basis_npa1)})  "
    f"|  shell = NPA2 ({len(basis_npa2)})"
)
print(
    f"Config B — moment matrix = NPA2 ({len(basis_npa2)})  "
    f"|  shell = NPA1 ({len(basis_npa1)})   [standard NPA2 baseline]"
)
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
# Main loop — run both configs for every eigenstate
# ---------------------------------------------------------------------------

lbs_A, ubs_A = [], []
lbs_B, ubs_B = [], []

print(
    f"\n{'#':>3}  {'E':>9}  {'exact':>7}  "
    f"{'A [lb, ub]':>24}  {'wA':>6}  "
    f"{'B [lb, ub]':>24}  {'wB':>6}"
)
print("-" * 96)

for i, (E, obs) in enumerate(zip(evals, obs_exact)):

    # Config A: small moment matrix, rich shell
    resA = bound_observable_excited(
        basis_npa1, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=basis_npa2,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lbA, ubA = resA.lb, resA.ub
    wA = (ubA - lbA) if not (np.isinf(lbA) or np.isinf(ubA)) else float("inf")

    # Config B: rich moment matrix, small shell  (standard NPA2 baseline)
    resB = bound_observable_excited(
        basis_npa2, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=basis_npa1,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lbB, ubB = resB.lb, resB.ub
    wB = (ubB - lbB) if not (np.isinf(lbB) or np.isinf(ubB)) else float("inf")

    lbs_A.append(lbA); ubs_A.append(ubA)
    lbs_B.append(lbB); ubs_B.append(ubB)

    print(
        f"{i+1:>3}  E={E:+.4f}  obs={obs:+.4f}  "
        f"[{lbA:+.4f},{ubA:+.4f}]  {wA:.4f}  "
        f"[{lbB:+.4f},{ubB:+.4f}]  {wB:.4f}"
    )

print("-" * 96)
total_wA = sum(u - l for l, u in zip(lbs_A, ubs_A)
               if not (np.isinf(l) or np.isinf(u)))
total_wB = sum(u - l for l, u in zip(lbs_B, ubs_B)
               if not (np.isinf(l) or np.isinf(u)))
print(f"Total certified width — Config A: {total_wA:.4f}  |  Config B: {total_wB:.4f}")
winner = "A  (shell=NPA2 wins — shell is the bottleneck)" \
         if total_wA < total_wB else \
         "B  (moment=NPA2 wins — moment matrix is the bottleneck)"
print(f"Tighter overall: Config {winner}")

# ---------------------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------------------

np.savez(
    out_npz,
    evals     = evals,
    obs_exact = obs_exact,
    lbs_A     = np.array(lbs_A),
    ubs_A     = np.array(ubs_A),
    lbs_B     = np.array(lbs_B),
    ubs_B     = np.array(ubs_B),
)
meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
    "config_A": "moment=NPA1, shell=NPA2",
    "config_B": "moment=NPA2, shell=NPA1",
    "npa1_size": len(basis_npa1),
    "npa2_size": len(basis_npa2),
    "total_width_A": total_wA,
    "total_width_B": total_wB,
    "observable": "Z_0",
}
out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved arrays   → {out_npz}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

x = np.arange(n_states)

COLOR_A = "steelblue"   # moment=NPA1, shell=NPA2
COLOR_B = "green"       # moment=NPA2, shell=NPA1  (standard baseline)

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_title(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g},  δE={DELTA_E}\n"
    f"Certified ⟨Z₀⟩ — shell vs moment matrix diagnostic\n"
    f"Config A (blue):  moment=NPA1, shell=NPA2  |  "
    f"Config B (green): moment=NPA2, shell=NPA1\n"
    f"Total width — A: {total_wA:.4f}   B: {total_wB:.4f}",
    fontsize=10,
)

for j in x:
    lbA, ubA = float(lbs_A[j]), float(ubs_A[j])
    lbB, ubB = float(lbs_B[j]), float(ubs_B[j])

    # Config A bar (thick, behind)
    if not (np.isinf(lbA) or np.isinf(ubA)):
        ax.plot([j, j], [lbA, ubA], color=COLOR_A, lw=10,
                solid_capstyle="round", zorder=1, alpha=0.6)

    # Config B bar (thinner, in front)
    if not (np.isinf(lbB) or np.isinf(ubB)):
        ax.plot([j, j], [lbB, ubB], color=COLOR_B, lw=5,
                solid_capstyle="round", zorder=2, alpha=0.8)

    # Exact value
    ax.scatter(j, obs_exact[j], color="black", s=30, zorder=4)

ax.set_xlabel("eigenstate index (sorted by energy)", fontsize=10)
ax.set_ylabel("⟨Z₀⟩", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.2f}" for e in evals], rotation=45, fontsize=7)
ax.legend(handles=[
    Patch(color=COLOR_A, alpha=0.6, label=f"Config A: moment=NPA1, shell=NPA2  (w={total_wA:.3f})"),
    Patch(color=COLOR_B, alpha=0.8, label=f"Config B: moment=NPA2, shell=NPA1  (w={total_wB:.3f})"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                  markersize=6, label="exact ⟨Z₀⟩"),
], fontsize=9)

plt.tight_layout()
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"Plot saved     → {out_pdf}")
