"""
XX-TFIM excited-state certification: scalar proxy RBM vs PSD shell.

RBM selects K_ADD words using the scalar relaxation (eq. 12) as the
optimization objective — cheap SCS calls, no PSD shell block.

The same words are then certified three ways (Mosek, parallelised):
  Method 1 — NPA1 baseline   : moment = shell = NPA1, PSD
  Method 2 — scalar          : moment = NPA1, shell = aug, scalar_flag=True
  Method 3 — PSD shell       : moment = NPA1, shell = aug, scalar_flag=False

Comparison answers: do words found under the cheap scalar proxy
transfer to tighter PSD bounds?

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.scalar_proxy_comparison
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.pauli_logic import PauliWord, local_pauli

from new_cert_excited_states.energy_shell_sdp import bound_observable_excited
from optimize.rbm import RBMTrainer

N         = 10
J         = 1.0
h         = 0.5
g         = 1.05
BC        = "periodic"   # periodic BC required for translation symmetry
DELTA_E   = 0.15
MOSEK_TOL = 1e-8
SCS_EPS   = 1e-4

K_ADD    = 32
RBM_SEED = 42

N_CERT    = 30
N_WORKERS = max(1, min(8, cpu_count() - 1))

INFEASIBLE_WIDTH = 1e10

O_dict = {local_pauli(0, "z"): 1.0}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sym = SymmetryManager(N=N, use_real_basis=True, use_translation=True, use_mirror=True)

basis_npa1 = generate_npa_basis(N, k=1).words
basis_npa2 = generate_npa_basis(N, k=2).words
npa1_set   = set(basis_npa1)
pool       = [w for w in basis_npa2 if w not in npa1_set]
L          = len(pool)

RBM_STEPS = max(500, 10 * L // K_ADD)
RBM_DECAY = 1.0 - 1.0 / K_ADD

tag     = f"N{N}_g{int(g*100):03d}_{BC}_dE{int(DELTA_E*100):03d}_kadd{K_ADD}_steps{RBM_STEPS}_ncert{N_CERT}"
out_npz = OUTPUT_DIR / f"scalar_proxy_{tag}.npz"
out_png = OUTPUT_DIR / f"scalar_proxy_{tag}.png"

print(f"XX-TFIM  N={N}, J={J}, h={h}, g={g}, {BC} BC,  dE={DELTA_E}")
print(f"NPA1: {len(basis_npa1)} words  |  NPA2: {len(basis_npa2)} words")
print(f"Pool (NPA2\\NPA1): {L} words  |  K_ADD={K_ADD}  |  RBM_STEPS={RBM_STEPS}")
print(f"Workers: {N_WORKERS}  |  N_CERT={N_CERT}")
print("=" * 72)

H_dict  = xx_tfising_hamiltonian_dict(N, J=J, h=h, g=g, boundary=BC)
H_exact = xx_tfising_hamiltonian_exact(N, J=J, h=h, g=g, boundary=BC)

evals_raw, _ = H_exact.eigenstates()
evals_all    = np.sort(np.real(evals_raw))
n_states     = len(evals_all)
mid_idx      = n_states // 2
E_mid        = float(evals_all[mid_idx])

sample_idx = np.linspace(0, n_states - 1, N_CERT, dtype=int)
evals      = evals_all[sample_idx]

print(f"Eigenstates: {n_states}  |  E range: [{evals_all[0]:.3f}, {evals_all[-1]:.3f}]")
print(f"Middle eigenstate: idx={mid_idx}, E={E_mid:+.4f}")
print(f"Sampled {N_CERT} eigenstates\n")


def _words_from_mask(mask) -> List[PauliWord]:
    return [pool[int(i)] for i in np.flatnonzero(np.asarray(mask))]


def _cert_single(shell_basis, E, scalar_flag=False, solver="MOSEK") -> Tuple[float, float]:
    t0  = time.time()
    res = bound_observable_excited(
        basis_npa1, H_dict, O_dict, E, DELTA_E, sym,
        shell_basis=shell_basis,
        use_commutation=True, scalar_flag=scalar_flag,
        mosek_tol=MOSEK_TOL, scs_eps=SCS_EPS, solver=solver,
    )
    elapsed = time.time() - t0
    lb, ub  = res.lb, res.ub
    if np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub):
        return float("nan"), elapsed
    return float(ub - lb), elapsed


# Module-level globals for multiprocessing workers (required for pickling on macOS).
_W_SHELL_BASIS  = None
_W_H_DICT       = None
_W_O_DICT       = None
_W_DELTA_E      = None
_W_SYM          = None
_W_MOSEK_TOL    = None
_W_SCALAR_FLAG  = None


def _worker_init(shell_basis, h_dict, o_dict, delta_e, sym_mgr, mosek_tol, scalar_flag):
    global _W_SHELL_BASIS, _W_H_DICT, _W_O_DICT
    global _W_DELTA_E, _W_SYM, _W_MOSEK_TOL, _W_SCALAR_FLAG
    _W_SHELL_BASIS  = shell_basis
    _W_H_DICT       = h_dict
    _W_O_DICT       = o_dict
    _W_DELTA_E      = delta_e
    _W_SYM          = sym_mgr
    _W_MOSEK_TOL    = mosek_tol
    _W_SCALAR_FLAG  = scalar_flag


def _worker_cert(E: float) -> Tuple[float, float]:
    t0  = time.time()
    res = bound_observable_excited(
        basis_npa1, _W_H_DICT, _W_O_DICT, E, _W_DELTA_E, _W_SYM,
        shell_basis=_W_SHELL_BASIS,
        use_commutation=True, scalar_flag=_W_SCALAR_FLAG,
        mosek_tol=_W_MOSEK_TOL, solver="MOSEK",
    )
    elapsed = time.time() - t0
    lb, ub  = res.lb, res.ub
    if np.isinf(lb) or np.isinf(ub) or np.isnan(lb) or np.isnan(ub):
        return float("nan"), elapsed
    return float(ub - lb), elapsed


def run_sweep_parallel(shell_basis, scalar_flag: bool, label: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"--- {label} ({N_CERT} states, {N_WORKERS} workers) {'─'*(40-len(label))}")
    t0       = time.time()
    initargs = (shell_basis, H_dict, O_dict, DELTA_E, sym, MOSEK_TOL, scalar_flag)
    with Pool(processes=N_WORKERS, initializer=_worker_init, initargs=initargs) as p:
        results = p.map(_worker_cert, evals.tolist())
    widths = np.array([r[0] for r in results])
    times  = np.array([r[1] for r in results])
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"total width={np.nansum(widths):.4f}  |  "
          f"mean time/state={np.nanmean(times):.2f}s\n")
    return widths, times


if __name__ == "__main__":

    print(f"--- RBM optimisation ({RBM_STEPS} steps, SCS, scalar proxy) {'─'*20}")

    def _obj_scalar(mask) -> float:
        shell = basis_npa1 + _words_from_mask(mask)
        w, _  = _cert_single(shell, E_mid, scalar_flag=True, solver="SCS")
        return w if not np.isnan(w) else INFEASIBLE_WIDTH

    trainer = RBMTrainer(
        obj_func=_obj_scalar, N=L, hamming_weight=K_ADD,
        seed=RBM_SEED, initial_guess=None,
    )
    trainer.decay = RBM_DECAY
    trainer.train(num_steps=RBM_STEPS, verbose=True)

    aug_basis = basis_npa1 + _words_from_mask(np.flatnonzero(trainer.current_vec))
    opt_cost  = float(trainer.current_cost)
    print(f"  best gap (scalar, SCS)={opt_cost:.6f}  |  shell size={len(aug_basis)}\n")

    opt_save = out_npz.with_name(out_npz.stem + "_opt.json")
    opt_save.write_text(json.dumps({
        "cost_scalar_scs": opt_cost,
        "shell_size": len(aug_basis),
        "mask": np.flatnonzero(trainer.current_vec).tolist(),
    }, indent=2))
    print(f"  Saved -> {opt_save}\n")

    widths_npa1,   times_npa1   = run_sweep_parallel(basis_npa1, False, "NPA1 baseline (PSD)")
    widths_scalar, times_scalar = run_sweep_parallel(aug_basis,  True,  "Scalar proxy  (aug, scalar)")
    widths_psd,    times_psd    = run_sweep_parallel(aug_basis,  False, "PSD shell     (aug, PSD)")

    np.savez(out_npz,
             evals=evals, sample_idx=sample_idx,
             widths_npa1=widths_npa1,   widths_scalar=widths_scalar,   widths_psd=widths_psd,
             times_npa1=times_npa1,     times_scalar=times_scalar,     times_psd=times_psd)

    meta = {
        "N": N, "J": J, "h": h, "g": g, "BC": BC, "delta_E": DELTA_E,
        "k_add": K_ADD, "rbm_steps": RBM_STEPS, "rbm_seed": RBM_SEED,
        "opt_solver": "SCS", "cert_solver": "MOSEK",
        "rbm_objective": "scalar_flag=True",
        "n_cert": N_CERT, "n_workers": N_WORKERS,
        "mid_idx": int(mid_idx), "E_mid": E_mid,
        "opt": {"cost_scalar_scs": opt_cost, "shell_size": len(aug_basis)},
        "total_width_npa1":   float(np.nansum(widths_npa1)),
        "total_width_scalar": float(np.nansum(widths_scalar)),
        "total_width_psd":    float(np.nansum(widths_psd)),
        "mean_time_npa1":     float(np.nanmean(times_npa1)),
        "mean_time_scalar":   float(np.nanmean(times_scalar)),
        "mean_time_psd":      float(np.nanmean(times_psd)),
    }
    out_npz.with_name(out_npz.stem + "_meta.json").write_text(json.dumps(meta, indent=2))

    COLOR_NPA1   = "tomato"
    COLOR_SCALAR = "darkorange"
    COLOR_PSD    = "mediumpurple"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={"hspace": 0.08})
    fig.suptitle(
        f"XX-TFIM  $N={N}$, $g={g}$, {BC} BC,  $\\delta E={DELTA_E}$  "
        f"[translation + mirror]\n"
        f"Scalar proxy RBM ({K_ADD} words, {RBM_STEPS} steps)  —  "
        f"do scalar-optimised words transfer to PSD?",
        fontsize=11,
    )

    ax1.scatter(evals, widths_npa1,   color=COLOR_NPA1,   s=10, alpha=0.8, zorder=3,
                label=f"NPA1 baseline  ({len(basis_npa1)} words, PSD)")
    ax1.scatter(evals, widths_scalar, color=COLOR_SCALAR, s=10, alpha=0.8, zorder=4,
                label=f"Scalar  ({len(aug_basis)} words, scalar)")
    ax1.scatter(evals, widths_psd,    color=COLOR_PSD,    s=10, alpha=0.8, zorder=5,
                label=f"PSD shell  ({len(aug_basis)} words, PSD)")
    ax1.set_ylabel("certification gap  (ub $-$ lb)", fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    ax2.scatter(evals, times_npa1,   color=COLOR_NPA1,   s=10, alpha=0.8, zorder=3,
                label=f"NPA1  (mean {np.nanmean(times_npa1):.2f}s)")
    ax2.scatter(evals, times_scalar, color=COLOR_SCALAR, s=10, alpha=0.8, zorder=4,
                label=f"Scalar  (mean {np.nanmean(times_scalar):.2f}s)")
    ax2.scatter(evals, times_psd,    color=COLOR_PSD,    s=10, alpha=0.8, zorder=5,
                label=f"PSD shell  (mean {np.nanmean(times_psd):.2f}s)")
    ax2.set_xlabel("eigenvalue $E$", fontsize=11)
    ax2.set_ylabel("runtime per eigenstate (s)", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out_png}")
    plt.show()
