"""
Correlation between magic (Stabilizer Rényi Entropy) and NPA certification gap.

For each eigenstate of the XX-TFIM (N=4):
  - Compute SRE of order 2:  M_2 = -log2( sum_P p_P^2 )
                              where p_P = <psi|P|psi>^2 / 2^N
  - Compute NPA2 certification gap (moment=NPA2, shell=NPA1)

Then plot SRE and gap vs eigenstate index, and a scatter of gap vs SRE.
A positive correlation would suggest magic is related to certification hardness.

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.magic_vs_cert_gap
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag     = f"N{N}_dE{int(DELTA_E * 100):03d}"
out_npz = OUTPUT_DIR / f"magic_vs_cert_gap_{tag}.npz"
out_pdf = out_npz.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

sym          = SymmetryManager(N=N, use_real_basis=True)
basis_npa1   = generate_npa_basis(N, k=1).words
basis_npa2   = generate_npa_basis(N, k=2).words

# Single-qubit Paulis (I, X, Y, Z)
_PAULIS = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]

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
evecs     = [evecs[i] for i in order]

n_states = len(evals)

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"Eigenstates: {n_states}  |  Pauli group size: {4**N}")
print("=" * 72)

# ---------------------------------------------------------------------------
# SRE computation
# ---------------------------------------------------------------------------

def compute_sre2(psi: qt.Qobj, N: int) -> float:
    """SRE of order 2 for a pure N-qubit state psi (qutip ket)."""
    sum_xi4 = 0.0
    for indices in itertools.product(range(4), repeat=N):
        P    = qt.tensor([_PAULIS[i] for i in indices])
        xi   = float(qt.expect(P, psi))   # real: P Hermitian, psi pure
        sum_xi4 += xi ** 4
    # p_P = xi^2 / 2^N  =>  sum_P p_P^2 = sum_P xi^4 / 4^N
    return float(-np.log2(sum_xi4 / 4**N))


print("\nComputing SRE for all eigenstates...")
sre = np.array([compute_sre2(evecs[i], N) for i in range(n_states)])
for i, (E, m) in enumerate(zip(evals, sre)):
    print(f"  {i+1:>2}  E={E:+.4f}  M_2={m:.4f}")

# ---------------------------------------------------------------------------
# NPA certification gaps  (moment=NPA2, shell=NPA1)
# ---------------------------------------------------------------------------

print("\nComputing NPA2 certification gaps (moment=NPA2, shell=NPA2)...")
gaps   = []
lbs    = []
ubs    = []

for i, E in enumerate(evals):
    res = bound_observable_excited(
        basis_npa2, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=basis_npa2,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lb, ub = res.lb, res.ub
    gap = (ub - lb) if not (np.isinf(lb) or np.isinf(ub)) else float("nan")
    gaps.append(gap); lbs.append(lb); ubs.append(ub)
    print(f"  {i+1:>2}  E={E:+.4f}  [{lb:+.4f},{ub:+.4f}]  gap={gap:.4f}")

gaps = np.array(gaps)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

np.savez(
    out_npz,
    evals=evals, obs_exact=obs_exact,
    sre=sre, gaps=gaps,
    lbs=np.array(lbs), ubs=np.array(ubs),
)
meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
    "moment_basis": "NPA2", "shell_basis": "NPA2",
    "sre_order": 2, "observable": "Z_0",
}
out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved → {out_npz}")

# Pearson correlation (ignoring NaN)
valid = ~np.isnan(gaps)
corr  = float(np.corrcoef(sre[valid], gaps[valid])[0, 1])
print(f"\nPearson correlation (SRE vs gap): r = {corr:+.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

x = np.arange(n_states)

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.5], wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

fig.suptitle(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g},  δE={DELTA_E}\n"
    f"Magic (SRE order 2) vs certification gap (moment=NPA2, shell=NPA2)  —  Pearson r={corr:+.3f}",
    fontsize=11,
)

# --- Panel 1: SRE per eigenstate ---
ax1.bar(x, sre, color="mediumpurple", alpha=0.8)
ax1.set_xlabel("eigenstate index", fontsize=10)
ax1.set_ylabel("SRE  $M_2$  (bits)", fontsize=10)
ax1.set_title("Magic per eigenstate", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{e:.2f}" for e in evals], rotation=45, fontsize=6)

# --- Panel 2: certification gap per eigenstate ---
gap_color = np.where(np.isnan(gaps), "lightgrey", "steelblue")
ax2.bar(x, np.nan_to_num(gaps), color=gap_color, alpha=0.8)
ax2.set_xlabel("eigenstate index", fontsize=10)
ax2.set_ylabel("certification gap  (ub − lb)", fontsize=10)
ax2.set_title("NPA2 gap per eigenstate", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{e:.2f}" for e in evals], rotation=45, fontsize=6)

# --- Panel 3: scatter gap vs SRE ---
ax3.scatter(sre[valid], gaps[valid], color="steelblue", s=50, zorder=3)
for i in np.where(valid)[0]:
    ax3.annotate(str(i + 1), (sre[i], gaps[i]),
                 textcoords="offset points", xytext=(4, 2), fontsize=7)

# Trend line
if valid.sum() > 2:
    m, b = np.polyfit(sre[valid], gaps[valid], 1)
    xlim = np.array([sre[valid].min(), sre[valid].max()])
    ax3.plot(xlim, m * xlim + b, color="tomato", lw=1.5, ls="--", zorder=2)

ax3.set_xlabel("SRE  $M_2$  (bits)", fontsize=10)
ax3.set_ylabel("certification gap", fontsize=10)
ax3.set_title(f"Scatter  (r={corr:+.3f})", fontsize=10)

plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"Plot saved → {out_pdf}")
