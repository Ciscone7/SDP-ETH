"""
Backbone synergy decomposition for excited-state shell optimisation — sampling.

For a single non-degenerate eigenstate of the XX-TFIM (N=8), scans
k = 1, ..., K_MAX additional shell words (from the NPA2\\NPA1 pool).
At each k, draws N_SAMPLES random k-word subsets uniformly at random and
computes for each sample S:

  Phi(S)   = f(∅) − f(S)            gap reduction
  Synergy(S) = Phi(S) - mean_{i in S}[ Phi(S \\ {i}) ]
               > 0  the k words work together (synergistic)
               < 0  some words are redundant given the others
               ≈ 0  the words contribute independently

Moment matrix is held fixed at NPA1 throughout (shell-only).
Each sample costs k+1 SDP calls, so total calls ≈ N_SAMPLES * sum_{k=1}^{K_MAX}(k+1).

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.excited_synergy_scan
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.pauli_logic import local_pauli

from new_cert_excited_states.energy_shell_sdp import bound_observable_excited
from optimize.synergy import calculate_synergy

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N         = 4
J         = 1.0
h         = 0.5
g         = 1.05
BC        = "open"
DELTA_E   = 0.15
MOSEK_TOL = 1e-8

TARGET_IDX = 7        # eigenstate index (0-based, sorted by energy); middle of N=4 spectrum (16 states)

K_MAX      = 54       # scan up to this k (= full pool size for N=4)
K_STEP     = 10       # step size: k = K_STEP, 2*K_STEP, ..., K_MAX
N_SAMPLES  = 10       # random subsets per k level
SEED       = 42

INFEASIBLE_WIDTH = 1e10

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag     = f"N{N}_g{int(g*100):03d}_{BC}_idx{TARGET_IDX}_kmax{K_MAX}_kstep{K_STEP}_ns{N_SAMPLES}"
out_npz = OUTPUT_DIR / f"excited_synergy_{tag}.npz"
out_png = OUTPUT_DIR / f"excited_synergy_{tag}.png"

# ---------------------------------------------------------------------------
# Bases and pool
# ---------------------------------------------------------------------------

np.random.seed(SEED)

sym        = SymmetryManager(N=N, use_real_basis=True)
basis_npa1 = generate_npa_basis(N, k=1).words
basis_npa2 = generate_npa_basis(N, k=2).words

npa1_set = set(basis_npa1)
pool     = [w for w in basis_npa2 if w not in npa1_set]   # 252 words
L        = len(pool)

# ---------------------------------------------------------------------------
# Exact diagonalisation — pick target eigenstate
# ---------------------------------------------------------------------------

H_dict  = xx_tfising_hamiltonian_dict(N, J=J, h=h, g=g, boundary=BC)
H_exact = xx_tfising_hamiltonian_exact(N, J=J, h=h, g=g, boundary=BC)

evals_raw, _ = H_exact.eigenstates()
evals        = np.sort(np.real(evals_raw))
E_target     = float(evals[TARGET_IDX])

gap_below = float(evals[TARGET_IDX] - evals[TARGET_IDX - 1])
gap_above = float(evals[TARGET_IDX + 1] - evals[TARGET_IDX])

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"Target eigenstate: idx={TARGET_IDX}, E={E_target:+.6f}")
print(f"  gap to neighbour below: {gap_below:.5f}")
print(f"  gap to neighbour above: {gap_above:.5f}")
ks_scan = list(range(K_STEP, K_MAX + 1, K_STEP))
print(f"Pool: {L} words (NPA2\\NPA1)  |  k levels: {ks_scan}  |  N_SAMPLES={N_SAMPLES}")
total_sdp = N_SAMPLES * sum(k + 1 for k in ks_scan)
print(f"Total SDP calls (excl. baseline): ~{total_sdp}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Objective: shell-only width (moment matrix fixed at NPA1)
# ---------------------------------------------------------------------------

def width_from_mask(mask) -> float:
    indices = np.flatnonzero(np.asarray(mask))
    shell   = basis_npa1 + [pool[int(i)] for i in indices]
    res = bound_observable_excited(
        basis_npa1, H_dict, O_dict, E_target, DELTA_E,
        sym, shell_basis=shell,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lb, ub = res.lb, res.ub
    if np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub):
        return INFEASIBLE_WIDTH
    return float(ub - lb)

def mask_from_indices(indices: np.ndarray) -> np.ndarray:
    m = np.zeros(L, dtype=int)
    m[indices] = 1
    return m

# f(∅): NPA1 baseline
f_zero = width_from_mask(np.zeros(L, dtype=int))
print(f"\nk=0  (NPA1 baseline)  width={f_zero:.6f}\n")

# ---------------------------------------------------------------------------
# Synergy scan — sampling
# ---------------------------------------------------------------------------

# results[k] = {"phis": array(N_SAMPLES,), "synergies": array(N_SAMPLES,)}
results = {}

for k in ks_scan:
    t0 = time.time()
    print(f"k={k}  ({N_SAMPLES} samples, {k+1} SDPs each) ...")

    phis_k      = np.empty(N_SAMPLES)
    synergies_k = np.empty(N_SAMPLES)

    for s in range(N_SAMPLES):
        idx   = np.random.choice(L, size=k, replace=False)
        mask  = mask_from_indices(idx)
        phi   = f_zero - width_from_mask(mask)
        syn   = calculate_synergy(width_from_mask, mask, k, f_zero=f_zero)
        phis_k[s]      = phi
        synergies_k[s] = syn

        if (s + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{s+1:3d}/{N_SAMPLES}]  {elapsed:.0f}s  "
                  f"mean_phi={phis_k[:s+1].mean():.4f}  "
                  f"mean_syn={synergies_k[:s+1].mean():.4f}")

    results[k] = {"phis": phis_k, "synergies": synergies_k}

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  |  "
          f"Φ: {phis_k.mean():.4f} ± {phis_k.std():.4f}  |  "
          f"synergy: {synergies_k.mean():.4f} ± {synergies_k.std():.4f}\n")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

save_dict = {"f_zero": np.array([f_zero])}
for k in ks_scan:
    save_dict[f"phis_{k}"]      = results[k]["phis"]
    save_dict[f"synergies_{k}"] = results[k]["synergies"]

np.savez(out_npz, **save_dict)

meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
    "target_idx": TARGET_IDX, "E_target": E_target,
    "gap_below": gap_below, "gap_above": gap_above,
    "K_MAX": K_MAX, "N_SAMPLES": N_SAMPLES, "seed": SEED,
    "f_zero": f_zero,
    "summary": {
        str(k): {
            "phi_mean":  float(results[k]["phis"].mean()),
            "phi_std":   float(results[k]["phis"].std()),
            "syn_mean":  float(results[k]["synergies"].mean()),
            "syn_std":   float(results[k]["synergies"].std()),
        }
        for k in ks_scan
    },
}
out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

ks = ks_scan

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                         gridspec_kw={"hspace": 0.1})

fig.suptitle(
    f"XX-TFIM  $N={N}$, $g={g}$, {BC} BC,  $\\delta E={DELTA_E}$\n"
    f"Synergy distribution — eigenstate {TARGET_IDX},  $E={E_target:+.4f}$"
    f"  ({N_SAMPLES} random samples per $k$)",
    fontsize=11,
)

# Top panel: Phi distribution
ax1 = axes[0]
ax1.boxplot([results[k]["phis"] for k in ks], positions=ks,
            widths=0.5, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.6),
            medianprops=dict(color="navy", lw=1.5),
            flierprops=dict(marker=".", markersize=3, alpha=0.4))
ax1.axhline(0, color="black", lw=0.8, ls="--")
ax1.set_ylabel(r"$\Phi(S) = f(\emptyset) - f(S)$", fontsize=11)
ax1.set_title("Gap reduction — distribution over random $k$-word shells", fontsize=10)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

# Bottom panel: Synergy distribution
ax2 = axes[1]
bp = ax2.boxplot([results[k]["synergies"] for k in ks], positions=ks,
                 widths=0.5, patch_artist=True,
                 medianprops=dict(color="darkred", lw=1.5),
                 flierprops=dict(marker=".", markersize=3, alpha=0.4))

for patch, k in zip(bp["boxes"], ks):
    mean_syn = results[k]["synergies"].mean()
    patch.set_facecolor("seagreen" if mean_syn >= 0 else "tomato")
    patch.set_alpha(0.6)

ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Synergy", fontsize=11)
ax2.set_xlabel("$k$  (shell words added)", fontsize=11)
ax2.set_title(r"Synergy $= \Phi(S) - \langle \Phi(S \setminus \{i\}) \rangle_i$"
              "  — distribution over random shells", fontsize=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
ax2.set_xticks(ks)

plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_png}")
plt.show()
