"""
Certification gap scan across the spectrum — XX-TFIM, NPA hierarchy + RBM basis selection.

WHAT THIS SCRIPT DOES
---------------------
For a 1D XX transverse-field Ising model (XX-TFIM) with N spins, this script
computes a rigorous SDP-based upper and lower bound on the expectation value of
a local observable (here Z_0) for every eigenstate in the spectrum, using the
NPA (Navascués–Pironio–Acín) hierarchy.

The NPA hierarchy certifies whether a given set of expectation values is
compatible with quantum mechanics, given only the Hamiltonian and an energy
shell constraint [E − δE/2, E + δE/2].  A tight bound means the eigenstate
behavior is tightly constrained (compatible with ETH); a loose bound means the
constraint is weak (hierarchy level too low or energy window too wide).

The script compares up to four methods:

  Method 1 — NPA1          : baseline; moment matrix = shell basis = NPA1 (~25 words)
  Method 2 — NPA1 + RBM    : moment matrix fixed at NPA1; shell augmented with
                              K_ADD words chosen by an RBM optimizer (REINFORCE).
                              The RBM optimises for a single target eigenstate
                              (default: the middle of the spectrum) and the
                              resulting shell is reused for ALL eigenstates.
  Method 3 — NPA1 + random : same augmentation size as Method 2 but words drawn
                              uniformly at random — sanity check for the RBM.
  Method 4 — NPA2          : full NPA level-2 basis (~270 words); very slow.

Set ONLY_OPT = True to run only the RBM optimisation + Method 2 certification
(useful when you just want the optimised basis and its certified gaps).

THE MODEL
---------
XX-TFIM:  H = J Σ (X_i X_{i+1} + Y_i Y_{i+1}) + h Σ Z_i + g Σ X_i
Parameters J, h, g are set near the Kim–Huse chaotic point (g ≈ 1.05, h = 0.5).
Boundary conditions (open / periodic) are controlled by BC.

KEY PARAMETERS  (edit at the top of the script)
------------------------------------------------
  N          — number of spins (start small: N=4 for testing, N=8 for results)
  g          — transverse field; g ≈ 1.05 gives ETH/chaotic regime
  BC         — "open" or "periodic"
  DELTA_E    — energy shell half-width δE (in units of J); try 0.1–0.3
  K_ADD      — how many extra Pauli words to add on top of NPA1 (e.g. 8–32)
  RBM_STEPS  — REINFORCE gradient steps for the RBM optimiser
  RUN_NPA2   — set False to skip the expensive NPA2 run
  ONLY_OPT   — set True to run only RBM opt + RBM-shell certification

OUTPUTS  (written to scripts/python/_outputs/)
----------------------------------------------
  spectrum_npa_comparison_<tag>.npz       — gaps arrays for all methods
  spectrum_npa_comparison_<tag>_meta.json — run parameters + summary stats
  spectrum_npa_comparison_<tag>_rbm_words.json — the K_ADD Pauli words chosen
                                                  by the RBM and their pool indices
  spectrum_npa_comparison_<tag>.png       — scatter plot: gap vs eigenvalue

HOW TO RUN
----------
  cd scripts/python
  python -m new_cert_excited_states.spectrum_npa_comparison

Requires: cvxpy, mosek (or scs as fallback), qutip, numpy, matplotlib.
See requirements.txt in this folder for pinned versions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.pauli_logic import local_pauli

from new_cert_excited_states.energy_shell_sdp import bound_observable_excited
from optimize.rbm import RBMTrainer

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N         = 4
J         = 1.0
h         = 0.5
g         = 1.05    # Kim & Huse chaotic regime
BC        = "open"
DELTA_E   = 0.15
MOSEK_TOL = 1e-8

K_ADD       = 8       # words to add on top of NPA1
RBM_STEPS   = 200
RBM_SEED    = 42
RBM_T_START = 1.0
RBM_ALPHA   = 0.99
RBM_DECAY   = 0.98

RUN_NPA2    = True    # set False to skip the slow NPA2 run
ONLY_OPT    = True    # set True to run only RBM opt + RBM-shell certification

INFEASIBLE_WIDTH = 1e10

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

tag     = f"N{N}_g{int(g*100):03d}_{BC}_dE{int(DELTA_E*100):03d}_kadd{K_ADD}_steps{RBM_STEPS}"
out_npz = OUTPUT_DIR / f"spectrum_npa_comparison_{tag}.npz"
out_png = OUTPUT_DIR / f"spectrum_npa_comparison_{tag}.png"

# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

# Mirror (reflection i → N−1−i) is valid for any uniform chain.
# Translation (cyclic shift) requires periodic BC.
sym = SymmetryManager(
    N=N,
    use_mirror=True,
    use_translation=(BC == "periodic"),
    use_real_basis=True,
)

basis_npa1 = generate_npa_basis(N, k=1).words
basis_npa2 = generate_npa_basis(N, k=2).words

npa1_set    = set(basis_npa1)
adding_pool = [w for w in basis_npa2 if w not in npa1_set]   # NPA2\NPA1, 252 words

sym_flags = "mirror" + (", translation" if BC == "periodic" else "")
print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"Symmetries: {sym_flags}")
print(f"NPA1: {len(basis_npa1)} words  |  NPA2: {len(basis_npa2)} words")
print(f"RBM pool (NPA2\\NPA1): {len(adding_pool)} words  |  k_add={K_ADD}")
print(f"RUN_NPA2={RUN_NPA2}  |  ONLY_OPT={ONLY_OPT}")
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
mid_idx  = n_states // 2

print(f"Eigenstates: {n_states}  |  E range: [{evals[0]:.3f}, {evals[-1]:.3f}]")
print(f"Mean level spacing: {(evals[-1]-evals[0])/(n_states-1):.4f}  |  δE={DELTA_E}")
print()

# ---------------------------------------------------------------------------
# Helper: run one SDP and return (lb, ub, width)
# ---------------------------------------------------------------------------

def _cert(basis, E):
    res = bound_observable_excited(
        basis, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=basis,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lb, ub = res.lb, res.ub
    if np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub):
        return float("inf"), float("-inf"), float("nan")
    return lb, ub, float(ub - lb)

# ---------------------------------------------------------------------------
# Pre-allocate all gap arrays (filled when the corresponding block runs)
# ---------------------------------------------------------------------------

gaps_npa1 = np.full(n_states, np.nan)
gaps_rbm  = np.full(n_states, np.nan)
gaps_rand = np.full(n_states, np.nan)
gaps_npa2 = np.full(n_states, np.nan)

# ---------------------------------------------------------------------------
# Method 1 — NPA1
# ---------------------------------------------------------------------------

if not ONLY_OPT:
    print("─── Method 1: NPA1 ─────────────────────────────────────────────────")
    t0 = time.time()
    for i, E in enumerate(evals):
        _, _, w = _cert(basis_npa1, E)
        gaps_npa1[i] = w
        if (i + 1) % 32 == 0:
            print(f"  [{i+1:3d}/{n_states}]  {time.time()-t0:.0f}s elapsed")
    print(f"  Done in {time.time()-t0:.1f}s  |  total width={np.nansum(gaps_npa1):.4f}\n")

# ---------------------------------------------------------------------------
# RBM — optimise for middle eigenstate, reuse basis for all
# ---------------------------------------------------------------------------

print(f"─── RBM optimisation (middle eigenstate, E={evals[mid_idx]:+.4f}) ──────────")

class WidthObjective:
    def __init__(self, E_center):
        self.E_center = E_center

    def __call__(self, mask):
        indices    = np.flatnonzero(np.asarray(mask))
        shell      = basis_npa1 + [adding_pool[int(i)] for i in indices]
        res = bound_observable_excited(
            basis_npa1, H_dict, O_dict, self.E_center, DELTA_E,
            sym, shell_basis=shell,
            use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
        )
        lb, ub = res.lb, res.ub
        if np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub):
            return INFEASIBLE_WIDTH
        return float(ub - lb)

t0      = time.time()
obj     = WidthObjective(E_center=evals[mid_idx])
trainer = RBMTrainer(
    obj_func=obj,
    N=len(adding_pool),
    hamming_weight=K_ADD,
    T_start=RBM_T_START,
    alpha=RBM_ALPHA,
    seed=RBM_SEED,
    initial_guess=None,
)
trainer.decay = RBM_DECAY
trainer.train(num_steps=RBM_STEPS, verbose=True)

rbm_mask   = np.asarray(trainer.current_vec)
rbm_indices = np.flatnonzero(rbm_mask).tolist()
rbm_shell  = basis_npa1 + [adding_pool[int(j)] for j in rbm_indices]
rbm_cost   = float(trainer.current_cost)
print(f"  Done in {time.time()-t0:.1f}s  |  best gap={rbm_cost:.6f}  "
      f"|  shell size={len(rbm_shell)}\n")

rbm_words_save = out_npz.with_name(out_npz.stem + "_rbm_words.json")
rbm_words_save.write_text(json.dumps({
    "cost": rbm_cost,
    "shell_size": len(rbm_shell),
    "pool_indices": rbm_indices,
    "words": [str(adding_pool[j]) for j in rbm_indices],
}, indent=2))
print(f"  RBM words saved -> {rbm_words_save}\n")

# ---------------------------------------------------------------------------
# Method 2 — NPA1 + RBM
# ---------------------------------------------------------------------------

print("─── Method 2: NPA1+RBM ─────────────────────────────────────────────")
t0 = time.time()
for i, E in enumerate(evals):
    res = bound_observable_excited(
        basis_npa1, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=rbm_shell,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lb, ub = res.lb, res.ub
    gaps_rbm[i] = float(ub - lb) if not (np.isinf(lb) or np.isinf(ub)) else np.nan
    if (i + 1) % 32 == 0:
        print(f"  [{i+1:3d}/{n_states}]  {time.time()-t0:.0f}s elapsed")
print(f"  Done in {time.time()-t0:.1f}s  |  total width={np.nansum(gaps_rbm):.4f}\n")

# ---------------------------------------------------------------------------
# Random shell + Method 3 — NPA1 + Random (skipped when ONLY_OPT)
# ---------------------------------------------------------------------------

if not ONLY_OPT:
    rng          = np.random.default_rng(RBM_SEED)
    rand_indices = rng.choice(len(adding_pool), size=K_ADD, replace=False)
    rand_shell   = basis_npa1 + [adding_pool[int(j)] for j in rand_indices]
    print(f"Random shell: {K_ADD} words drawn at random (seed={RBM_SEED}), shell size={len(rand_shell)}\n")

    print("─── Method 3: NPA1+Random ──────────────────────────────────────────")
    t0 = time.time()
    for i, E in enumerate(evals):
        res = bound_observable_excited(
            basis_npa1, H_dict, O_dict, E, DELTA_E,
            sym, shell_basis=rand_shell,
            use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
        )
        lb, ub = res.lb, res.ub
        gaps_rand[i] = float(ub - lb) if not (np.isinf(lb) or np.isinf(ub)) else np.nan
        if (i + 1) % 32 == 0:
            print(f"  [{i+1:3d}/{n_states}]  {time.time()-t0:.0f}s elapsed")
    print(f"  Done in {time.time()-t0:.1f}s  |  total width={np.nansum(gaps_rand):.4f}\n")

# ---------------------------------------------------------------------------
# Method 4 — NPA2  (optional, skipped when ONLY_OPT)
# ---------------------------------------------------------------------------

if not ONLY_OPT and RUN_NPA2:
    print("─── Method 4: NPA2 ─────────────────────────────────────────────────")
    t0 = time.time()
    for i, E in enumerate(evals):
        _, _, w = _cert(basis_npa2, E)
        gaps_npa2[i] = w
        if (i + 1) % 16 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_states - i - 1)
            print(f"  [{i+1:3d}/{n_states}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")
    print(f"  Done in {time.time()-t0:.1f}s  |  total width={np.nansum(gaps_npa2):.4f}\n")
elif not ONLY_OPT:
    print("─── Method 4: NPA2  SKIPPED (RUN_NPA2=False) ───────────────────────\n")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

np.savez(out_npz,
         evals=evals, obs_exact=obs_exact,
         gaps_npa1=gaps_npa1, gaps_rbm=gaps_rbm,
         gaps_rand=gaps_rand, gaps_npa2=gaps_npa2)

meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
    "k_add": K_ADD, "rbm_steps": RBM_STEPS, "rbm_seed": RBM_SEED,
    "rbm_alpha": RBM_ALPHA, "rbm_decay": RBM_DECAY,
    "mid_eigenstate_energy": float(evals[mid_idx]),
    "rbm_best_gap": rbm_cost,
    "rbm_shell_size": len(rbm_shell),
    "rand_shell_indices": rand_indices.tolist() if not ONLY_OPT else None,
    "optimized": "shell_only",
    "only_opt": ONLY_OPT,
    "npa2_run": RUN_NPA2 and not ONLY_OPT,
    "total_width_npa1": float(np.nansum(gaps_npa1)) if not ONLY_OPT else None,
    "total_width_rbm":  float(np.nansum(gaps_rbm)),
    "total_width_rand": float(np.nansum(gaps_rand)) if not ONLY_OPT else None,
    "total_width_npa2": float(np.nansum(gaps_npa2)) if (not ONLY_OPT and RUN_NPA2) else None,
}
out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLOR_NPA1 = "tomato"
COLOR_RBM  = "mediumpurple"
COLOR_RAND = "darkorange"
COLOR_NPA2 = "steelblue"

fig, ax = plt.subplots(figsize=(13, 5))

if not ONLY_OPT:
    ax.scatter(evals, gaps_npa1, color=COLOR_NPA1, s=18, alpha=0.8, zorder=3,
               label=f"NPA1  ({len(basis_npa1)} words)")
    ax.scatter(evals, gaps_rand, color=COLOR_RAND, s=18, alpha=0.8, zorder=4,
               label=f"NPA1+{K_ADD} random shell  ({len(rand_shell)} words)")
ax.scatter(evals, gaps_rbm,  color=COLOR_RBM,  s=18, alpha=0.8, zorder=5,
           label=f"NPA1+{K_ADD} RBM shell  ({len(rbm_shell)} words)")
if not ONLY_OPT and RUN_NPA2:
    ax.scatter(evals, gaps_npa2, color=COLOR_NPA2, s=18, alpha=0.8, zorder=6,
               label=f"NPA2  ({len(basis_npa2)} words)")

ax.set_xlabel("eigenvalue $E$", fontsize=12)
ax.set_ylabel("certification gap  (ub $-$ lb)", fontsize=12)
ax.set_title(
    f"XX-TFIM  $N={N}$, $J={J}$, $h={h}$, $g={g}$, {BC} BC,  $\\delta E={DELTA_E}$\n"
    f"Certification gap on $\\langle Z_0 \\rangle$ — NPA1 vs NPA1+{K_ADD}(random) vs NPA1+{K_ADD}(RBM) vs NPA2",
    fontsize=11,
)
ax.set_ylim(bottom=0)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Plot saved → {out_png}")
plt.show()
