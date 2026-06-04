"""
Benchmark: moment matrix fixed at NPA2, shell basis at NPA1 / NPA2 / NPA3 / NPA4.

Shows the convergence of certified bounds on <Z_0> as the shell basis grows,
with the moment matrix held constant. This establishes:

  - The current baseline  (shell=NPA1, what the RBM starts from)
  - The ceiling           (shell=NPA4, best achievable with this moment matrix)
  - How much room the RBM has to close

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.shell_hierarchy_benchmark
"""

from __future__ import annotations

import json
import time
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
g         = 1.05   # Kim & Huse (2013) chaotic regime
BC        = "open"
DELTA_E   = 0.15
MOSEK_TOL = 1e-8

SHELL_LEVELS = [1, 2, 3, 4]   # NPA levels for the shell basis

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag     = f"N{N}_dE{int(DELTA_E * 100):03d}_moment_npa2"
out_npz = OUTPUT_DIR / f"shell_hierarchy_benchmark_{tag}.npz"
out_pdf = out_npz.with_suffix(".pdf")

# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

sym          = SymmetryManager(N=N, use_real_basis=True)
moment_basis = generate_npa_basis(N, k=2).words   # fixed throughout

shell_bases  = {}
for lv in SHELL_LEVELS:
    shell_bases[lv] = generate_npa_basis(N, k=lv).words

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"Moment matrix: NPA2 ({len(moment_basis)} words)  —  fixed")
for lv in SHELL_LEVELS:
    print(f"  Shell NPA{lv}: {len(shell_bases[lv])} words")
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
# Main loop
# ---------------------------------------------------------------------------

results = {lv: {"lbs": [], "ubs": []} for lv in SHELL_LEVELS}

for lv in SHELL_LEVELS:
    shell = shell_bases[lv]
    t0    = time.time()
    print(f"\nShell = NPA{lv} ({len(shell)} words)")
    print(f"  {'#':>3}  {'E':>9}  {'exact':>7}  {'[lb, ub]':>24}  {'width':>7}")
    print(f"  {'-'*58}")

    for i, (E, obs) in enumerate(zip(evals, obs_exact)):
        res = bound_observable_excited(
            moment_basis, H_dict, O_dict, E, DELTA_E,
            sym, shell_basis=shell,
            use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
        )
        lb, ub = res.lb, res.ub
        w = (ub - lb) if not (np.isinf(lb) or np.isinf(ub)) else float("inf")
        results[lv]["lbs"].append(lb)
        results[lv]["ubs"].append(ub)
        print(f"  {i+1:>3}  E={E:+.4f}  obs={obs:+.4f}  [{lb:+.4f},{ub:+.4f}]  {w:.4f}")

    total_w = sum(
        u - l for l, u in zip(results[lv]["lbs"], results[lv]["ubs"])
        if not (np.isinf(l) or np.isinf(u))
    )
    elapsed = time.time() - t0
    print(f"  Total width: {total_w:.4f}   ({elapsed:.1f} s)")
    results[lv]["total_width"] = total_w
    results[lv]["elapsed"]     = elapsed

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print(f"{'Shell':>8}  {'words':>6}  {'total width':>12}  {'time (s)':>9}")
print("-" * 42)
for lv in SHELL_LEVELS:
    print(
        f"  NPA{lv}  {len(shell_bases[lv]):>6}  "
        f"{results[lv]['total_width']:>12.4f}  "
        f"{results[lv]['elapsed']:>9.1f}"
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

save_dict = {"evals": evals, "obs_exact": obs_exact}
for lv in SHELL_LEVELS:
    save_dict[f"lbs_shell{lv}"] = np.array(results[lv]["lbs"])
    save_dict[f"ubs_shell{lv}"] = np.array(results[lv]["ubs"])
np.savez(out_npz, **save_dict)

meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
    "moment_basis": "NPA2",
    "moment_basis_size": len(moment_basis),
    "shell_levels": SHELL_LEVELS,
    "shell_sizes": {lv: len(shell_bases[lv]) for lv in SHELL_LEVELS},
    "total_widths": {lv: results[lv]["total_width"] for lv in SHELL_LEVELS},
    "observable": "Z_0",
}
out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nSaved arrays   → {out_npz}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLORS = {1: "tomato", 2: "orange", 3: "steelblue", 4: "seagreen"}

x = np.arange(n_states)

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g},  δE={DELTA_E}\n"
    f"Certified ⟨Z₀⟩ — moment=NPA2 (fixed), shell hierarchy NPA1→NPA4\n"
    + "  ".join(
        f"NPA{lv}: w={results[lv]['total_width']:.3f}"
        for lv in SHELL_LEVELS
    ),
    fontsize=10,
)

lw_map = {1: 14, 2: 10, 3: 6, 4: 3}   # thicker behind, thinner in front

for lv in SHELL_LEVELS:
    lbs = results[lv]["lbs"]
    ubs = results[lv]["ubs"]
    for j in x:
        lb, ub = float(lbs[j]), float(ubs[j])
        if not (np.isinf(lb) or np.isinf(ub)):
            ax.plot([j, j], [lb, ub],
                    color=COLORS[lv], lw=lw_map[lv],
                    solid_capstyle="round",
                    zorder=lv, alpha=0.75)

for j in x:
    ax.scatter(j, obs_exact[j], color="black", s=30, zorder=10)

ax.set_xlabel("eigenstate index (sorted by energy)", fontsize=10)
ax.set_ylabel("⟨Z₀⟩", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.2f}" for e in evals], rotation=45, fontsize=7)

legend_handles = [
    Patch(color=COLORS[lv], alpha=0.75,
          label=f"shell=NPA{lv}  ({len(shell_bases[lv])} words,  w={results[lv]['total_width']:.3f})")
    for lv in SHELL_LEVELS
] + [
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                  markersize=6, label="exact ⟨Z₀⟩")
]
ax.legend(handles=legend_handles, fontsize=9)

plt.tight_layout()
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"Plot saved     → {out_pdf}")
