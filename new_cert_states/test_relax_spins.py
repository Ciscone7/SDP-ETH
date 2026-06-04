"""
NPA relaxation hierarchy for the TF Ising ground state energy.

For a 1D transverse-field Ising chain
    H = -J Σ Z_i Z_{i+1}  -  h Σ X_i  -  k Σ Z_i
computes the exact ground state energy and the NPA lower bound at levels
1, 2, 3, 4.  Plots the relaxation gap  E_exact − E_NPA(k)  vs  k.

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_states.test_relax_spins
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from spins.basis_builder import generate_npa_basis
from spins.models import ising_hamiltonian_exact, ising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.spins_sdp import solve_pauli_relaxation

# ---------------------------------------------------------------------------
# Parameters  (edit freely)
# ---------------------------------------------------------------------------

N          = 8        # number of spins
J          = 1.0      # ZZ coupling
h          = 1.0      # transverse field (X)
k_field    = 0.0      # longitudinal field (Z)
BC         = "open"   # "open" or "periodic"
MOSEK_TOL  = 1e-8

NPA_LEVELS = [1, 2, 3]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag     = f"tfising_N{N}_J{int(J*10):02d}_h{int(h*10):02d}_{BC}"
out_png = OUTPUT_DIR / f"test_relax_spins_{tag}.png"

# ---------------------------------------------------------------------------
# Exact diagonalisation
# ---------------------------------------------------------------------------

H_qt    = ising_hamiltonian_exact(N, J=J, h=h, k=k_field, boundary=BC)
H_dict  = ising_hamiltonian_dict(N, J=J, h=h, k=k_field, boundary=BC)

evals   = np.sort(np.real(H_qt.eigenenergies()))
E_exact = float(evals[0])

print(f"TF Ising  N={N}, J={J}, h={h}, k={k_field}, {BC} BC")
print(f"Exact ground state energy: E₀ = {E_exact:.8f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# NPA relaxation at each level
# ---------------------------------------------------------------------------

sym = SymmetryManager(N=N, use_real_basis=True)

npa_energies = {}
npa_sizes    = {}

for lv in NPA_LEVELS:
    basis = generate_npa_basis(N, k=lv).words
    npa_sizes[lv] = len(basis)
    t0 = time.time()
    E_npa = solve_pauli_relaxation(basis, H_dict, sym, sense="min", mosek_tol=MOSEK_TOL)
    elapsed = time.time() - t0
    npa_energies[lv] = E_npa
    gap = E_exact - E_npa
    print(f"  NPA{lv}  ({len(basis):4d} words)  E = {E_npa:.8f}  "
          f"gap = {gap:.2e}  ({elapsed:.1f} s)")

# ---------------------------------------------------------------------------
# Plot: relaxation gap vs NPA level
# ---------------------------------------------------------------------------

gaps = np.array([E_exact - npa_energies[lv] for lv in NPA_LEVELS])

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(NPA_LEVELS, gaps, marker="o", color="steelblue", lw=1.8, ms=7)

for lv, gap in zip(NPA_LEVELS, gaps):
    ax.annotate(
        f"{gap:.2e}",
        xy=(lv, gap), xytext=(6, 4), textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel("NPA level  $k$", fontsize=12)
ax.set_ylabel(r"$E_{\rm exact} - E_{\rm NPA}(k)$", fontsize=12)
ax.set_title(
    f"TF Ising  $N={N}$, $J={J}$, $h={h}$, {BC} BC\n"
    f"Ground-state relaxation gap vs NPA level",
    fontsize=11,
)
ax.set_xticks(NPA_LEVELS)
ax.set_xticklabels([f"NPA{lv}\n({npa_sizes[lv]} words)" for lv in NPA_LEVELS], fontsize=9)
ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.axhline(0, color="black", lw=0.8, ls="--")

plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_png}")
plt.show()
