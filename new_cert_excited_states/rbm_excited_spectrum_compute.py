"""
Compute certified bounds on <Z_0> for all eigenstates of the XX-TFIM (N=4).

Two methods compared:

  Baseline      — moment matrix = NPA2  |  shell = NPA2
  RBM-optimised — moment matrix = shell = NPA1 + k best words from NPA3\\NPA1
                  RBM minimises the joint bound width (ub - lb) by choosing
                  which k words to use for BOTH the moment matrix and shell.
                  Pool: NPA3\\NPA1 (162 words).  k = |NPA2\\NPA1| = 54.
                  Warm-start: NPA2\\NPA1 words (initial cost = NPA2 baseline).

OPTIMIZE_PER_EIGENSTATE controls how the RBM shell basis is chosen:

  True  — optimize independently for every eigenstate  (expensive, tightest)
  False — optimize once for the middle eigenstate, reuse that shell for all
          others  (cheap, tests transferability)

Results are saved to _outputs/rbm_excited_spectrum_<tag>.npz for plotting.

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.rbm_excited_spectrum_compute
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import qutip as qt

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
g         = 1.05   # Kim & Huse (2013) chaotic regime
BC        = "open"
DELTA_E   = 0.15
MOSEK_TOL = 1e-8

# RBM hyperparameters
RBM_STEPS    = 2000
RBM_SEED     = 42
RBM_T_START  = 1.0
RBM_ALPHA    = 0.99
RBM_DECAY    = 0.98   # baseline moving-average decay (set after RBMTrainer init)

# --- Mode flag ---
# True  : run the RBM independently for each eigenstate
# False : run the RBM once for the middle eigenstate, reuse the basis for all
OPTIMIZE_PER_EIGENSTATE = True

# --- Warm-start flag ---
# True  : initialise RBM mask to the NPA2\NPA1 words (guaranteed >= NPA2 quality)
# False : start from a random k-subset of the pool (may be worse than NPA2 early on)
WARM_START_NPA2 = True

INFEASIBLE_WIDTH = 1e10

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

_mode_tag = "per_state" if OPTIMIZE_PER_EIGENSTATE else "middle_only"
tag      = f"N{N}_g{int(g*100):03d}_{BC}_dE{int(DELTA_E*100):03d}_steps{RBM_STEPS}_seed{RBM_SEED}_{_mode_tag}_jointopt"
out_npz  = OUTPUT_DIR / f"rbm_excited_spectrum_{tag}.npz"
out_meta = out_npz.with_name(out_npz.stem + "_meta.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """Return '1st', '2nd', '3rd', '4th', ... for n >= 1."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th'][n % 10]}"


def _run_rbm(E_center: float) -> list:
    """Run the RBM for a given energy centre; return the best shell basis."""
    obj     = WidthObjective(E_center=E_center)
    trainer = RBMTrainer(
        obj_func=obj,
        N=len(adding_pool),
        hamming_weight=k,
        T_start=RBM_T_START,
        alpha=RBM_ALPHA,
        seed=RBM_SEED,
        initial_guess=npa2_initial_mask if WARM_START_NPA2 else None,
    )
    trainer.decay = RBM_DECAY
    trainer.train(num_steps=RBM_STEPS, verbose=False)
    mask = np.asarray(trainer.current_vec)
    return basis_npa1 + [adding_pool[int(j)] for j in np.flatnonzero(mask)]

# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

sym = SymmetryManager(N=N, use_real_basis=True)

basis_npa1 = generate_npa_basis(N, k=1).words
basis_npa2 = generate_npa_basis(N, k=2).words
basis_npa3 = generate_npa_basis(N, k=3).words

npa1_set    = set(basis_npa1)
adding_pool = [w for w in basis_npa3 if w not in npa1_set]
k           = sum(1 for w in basis_npa2 if w not in npa1_set)

# Warm-start mask: 1 at positions in adding_pool that belong to NPA2\NPA1
npa2_only          = set(basis_npa2) - npa1_set
npa2_initial_mask  = np.array([1 if w in npa2_only else 0 for w in adding_pool], dtype=int)

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  δE={DELTA_E}")
print(f"NPA1: {len(basis_npa1)}  |  NPA2: {len(basis_npa2)}  |  NPA3: {len(basis_npa3)}")
print(f"Moment matrix: NPA2 (fixed)  |  Shell pool (NPA3\\NPA1): {len(adding_pool)} words  |  k={k}")
print(f"RBM: {RBM_STEPS} steps,  seed={RBM_SEED},  alpha={RBM_ALPHA},  decay={RBM_DECAY}")
print(f"Warm-start: {'NPA2\\NPA1 shell words' if WARM_START_NPA2 else 'random'}")
print(f"Mode: {'per-eigenstate optimisation' if OPTIMIZE_PER_EIGENSTATE else 'middle-eigenstate shell reused for all'}")
print(f"Output: {out_npz}")
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

n_states   = len(evals)
mid_idx    = n_states // 2   # index of the middle eigenstate (0-based)

# ---------------------------------------------------------------------------
# RBM objective
# ---------------------------------------------------------------------------

class WidthObjective:
    def __init__(self, E_center: float) -> None:
        self.E_center = E_center

    def __call__(self, mask) -> float:
        indices = np.flatnonzero(np.asarray(mask))
        basis   = basis_npa1 + [adding_pool[int(i)] for i in indices]
        res = bound_observable_excited(
            basis, H_dict, O_dict, self.E_center, DELTA_E,
            sym, shell_basis=basis,
            use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
        )
        lb, ub = res.lb, res.ub
        if np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub):
            return INFEASIBLE_WIDTH
        return float(ub - lb)

# ---------------------------------------------------------------------------
# Pre-compute the shared basis when not optimising per eigenstate
# ---------------------------------------------------------------------------

shared_shell: list | None = None

if not OPTIMIZE_PER_EIGENSTATE:
    E_mid = evals[mid_idx]
    print(
        f"\nRunning RBM once for the {_ordinal(mid_idx + 1)} eigenstate "
        f"(E = {E_mid:+.6f})  —  shell basis will be reused for all eigenstates."
    )
    shared_shell = _run_rbm(E_mid)
    print(f"RBM done.  Shared shell size: {len(shared_shell)} words.\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

lbs_npa2, ubs_npa2 = [], []
lbs_rbm,  ubs_rbm  = [], []

print(
    f"\n{'#':>3}  {'E':>9}  {'exact':>7}  "
    f"{'baseline [lb, ub]':>24}  {'wb':>6}  "
    f"{'RBM-shell [lb, ub]':>24}  {'w_r':>6}"
)
print("-" * 96)

for i, (E, obs) in enumerate(zip(evals, obs_exact)):

    # Baseline: moment=NPA2, shell=NPA2
    res2 = bound_observable_excited(
        basis_npa2, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=basis_npa2,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lb2, ub2 = res2.lb, res2.ub
    w2 = (ub2 - lb2) if not (np.isinf(lb2) or np.isinf(ub2)) else float("inf")

    # RBM-optimised shell — either fresh or shared
    if OPTIMIZE_PER_EIGENSTATE:
        print(
            f"  → optimising shell for {_ordinal(i + 1)} eigenstate  "
            f"(E = {E:+.6f}) ..."
        )
        best_shell = _run_rbm(E)
    else:
        best_shell = shared_shell

    # RBM result: moment=shell=optimised shared basis
    res_r  = bound_observable_excited(
        best_shell, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=best_shell,
        use_commutation=True, scalar_flag=False, mosek_tol=MOSEK_TOL,
    )
    lb_r, ub_r = res_r.lb, res_r.ub
    w_r = (ub_r - lb_r) if not (np.isinf(lb_r) or np.isinf(ub_r)) else float("inf")

    lbs_npa2.append(lb2);  ubs_npa2.append(ub2)
    lbs_rbm.append(lb_r);  ubs_rbm.append(ub_r)

    print(
        f"{i+1:>3}  E={E:+.4f}  obs={obs:+.4f}  "
        f"[{lb2:+.4f},{ub2:+.4f}]  {w2:.4f}  "
        f"[{lb_r:+.4f},{ub_r:+.4f}]  {w_r:.4f}"
    )

print("-" * 96)
total_w2 = sum(u - l for l, u in zip(lbs_npa2, ubs_npa2)
               if not (np.isinf(l) or np.isinf(u)))
total_wr = sum(u - l for l, u in zip(lbs_rbm,  ubs_rbm)
               if not (np.isinf(l) or np.isinf(u)))
print(f"Total certified width — NPA2: {total_w2:.4f}  |  RBM-opt: {total_wr:.4f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

np.savez(
    out_npz,
    evals     = evals,
    obs_exact = obs_exact,
    lbs_npa2  = np.array(lbs_npa2),
    ubs_npa2  = np.array(ubs_npa2),
    lbs_rbm   = np.array(lbs_rbm),
    ubs_rbm   = np.array(ubs_rbm),
)

meta = {
    "N": N, "J": J, "h": h, "g": g, "BC": BC,
    "delta_E": DELTA_E,
    "rbm_steps": RBM_STEPS,
    "rbm_seed": RBM_SEED,
    "rbm_alpha": RBM_ALPHA,
    "rbm_decay": RBM_DECAY,
    "rbm_T_start": RBM_T_START,
    "warm_start_npa2": WARM_START_NPA2,
    "optimize_per_eigenstate": OPTIMIZE_PER_EIGENSTATE,
    "opt_eigenstate_index": None if OPTIMIZE_PER_EIGENSTATE else int(mid_idx),
    "opt_eigenstate_energy": None if OPTIMIZE_PER_EIGENSTATE else float(evals[mid_idx]),
    "k": k,
    "pool_size": len(adding_pool),
    "npa1_size": len(basis_npa1),
    "npa2_size": len(basis_npa2),
    "npa3_size": len(basis_npa3),
    "observable": "Z_0",
    "optimized_target": "joint_moment_and_shell",
    "baseline_label": "moment=NPA2, shell=NPA2",
    "rbm_label": "moment=shell=NPA1+k_opt (joint)",
    "total_width_npa2": total_w2,
    "total_width_rbm":  total_wr,
}
out_meta.write_text(json.dumps(meta, indent=2))

print(f"\nSaved arrays  → {out_npz}")
print(f"Saved metadata → {out_meta}")
