"""
Validation for new_cert_excited_states/energy_shell_sdp.py.

Tests
-----
1. Correctness: XX-TFIM, all eigenvalues, exact values inside certified bounds.
2. Tightness:   side-by-side bound widths for PSD blocks (F ≽ 0, G ≽ 0) vs
                scalar element-wise constraints (|⟨w_i w_j (H-E)⟩| ≤ δE).

Run:
    .venv/bin/python -m new_cert_excited_states.test_energy_shell
"""

import numpy as np
import qutip as qt

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.pauli_logic import local_pauli

from new_cert_excited_states.energy_shell_sdp import bound_observable_excited as bound_new

# ---------------------------------------------------------------------------
# Problem setup  —  TFIM N=3, open BC (no spatial symmetry needed)
# ---------------------------------------------------------------------------

N  = 4
J  = 1.0
h  = 0.5
g  = 0.3
BC = "periodic"

sym = SymmetryManager(N=N, use_real_basis=True)

basis_npa2 = generate_npa_basis(N, k=2).words   # main basis
shell_npa1 = generate_npa_basis(N, k=1).words   # shell basis for F, G

O_dict = {local_pauli(0, "z"): 1.0}

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

print(f"XX-TFIM N={N}, J={J}, h={h}, g={g}, {BC} BC")
print(f"Main basis: NPA2 ({len(basis_npa2)} words), shell: NPA1 ({len(shell_npa1)} words)")
print(f"{'='*72}")

# ---------------------------------------------------------------------------
# Run both implementations and compare
# ---------------------------------------------------------------------------

DELTA_E = 0.15

print(f"\n{'Eigenvalue':>12}  {'Exact <Z0>':>11}  "
      f"{'PSD [lb,ub]':>26}  {'w':>6}  "
      f"{'Scalar [lb,ub]':>26}  {'w':>6}")
print("-" * 95)

all_pass_new    = True
all_pass_scalar = True
total_width_new    = 0.0
total_width_scalar = 0.0
lbs_new, ubs_new = [], []
lbs_sc,  ubs_sc  = [], []

for i, (E, obs) in enumerate(zip(evals, obs_exact)):
    res_new = bound_new(
        basis_npa2, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=shell_npa1, use_commutation=True, scalar_flag=False, mosek_tol=1e-8,
    )
    res_scalar = bound_new(
        basis_npa2, H_dict, O_dict, E, DELTA_E,
        sym, shell_basis=shell_npa1, use_commutation=True, scalar_flag=True, mosek_tol=1e-8,
    )

    lb_new, ub_new = res_new.lb, res_new.ub
    lb_sc,  ub_sc  = res_scalar.lb, res_scalar.ub

    w_new    = ub_new - lb_new
    w_scalar = ub_sc  - lb_sc

    if not (lb_new - 1e-4 <= obs <= ub_new + 1e-4): all_pass_new    = False
    if not (lb_sc  - 1e-4 <= obs <= ub_sc  + 1e-4): all_pass_scalar = False

    total_width_new    += w_new
    total_width_scalar += w_scalar
    lbs_new.append(lb_new); ubs_new.append(ub_new)
    lbs_sc.append(lb_sc);   ubs_sc.append(ub_sc)

    print(f"E={E:8.4f}  obs={obs:+7.4f}  "
          f"[{lb_new:+7.4f},{ub_new:+7.4f}] {w_new:.4f}  "
          f"[{lb_sc:+7.4f},{ub_sc:+7.4f}] {w_scalar:.4f}")

print("-" * 95)
print(f"{'Total width':>50}  {total_width_new:.4f}    {total_width_scalar:.4f}")
psd_vs_scalar = (total_width_new - total_width_scalar) / total_width_new * 100
print(f"\nScalar vs PSD:      {psd_vs_scalar:+.1f}%  (negative = scalar tighter)")
print(f"\nPSD correctness:    {'ALL PASS' if all_pass_new    else 'SOME FAIL'}")
print(f"Scalar correctness: {'ALL PASS' if all_pass_scalar else 'SOME FAIL'}")

# ---------------------------------------------------------------------------
# Report free moment count
# ---------------------------------------------------------------------------

from new_cert_excited_states.energy_shell_sdp import _collect_free_moments, _make_ext_label_index
from spins.spins_sdp import build_block_reps

reps, _ = build_block_reps(basis_npa2, sym)
main_idx = reps[0].label_index
free_m   = _collect_free_moments(shell_npa1, H_dict, main_idx, sym)

print(f"\nFree shell moments introduced: {len(free_m)}")
print(f"Main y-vector size           : {len(main_idx)}")
print(f"Extended yz-vector size      : {len(main_idx) + len(free_m)}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

x      = np.arange(len(evals))
titles = ["PSD", "Scalar"]
data   = [(lbs_new, ubs_new), (lbs_sc, ubs_sc)]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle(f"XX-TFIM N={N}, J={J}, h={h}, g={g}, δE={DELTA_E}  —  bound width per eigenstate", fontsize=13)

for ax, title, (lbs, ubs) in zip(axes, titles, data):
    for i in x:
        if np.isinf(lbs[i]) or np.isinf(ubs[i]):
            dot_color, bar_color = "gold", "lemonchiffon"
        elif lbs[i] - 1e-4 <= obs_exact[i] <= ubs[i] + 1e-4:
            dot_color, bar_color = "green", "lightgreen"
        else:
            dot_color, bar_color = "red", "lightcoral"
        ax.plot([i, i], [lbs[i], ubs[i]], color=bar_color, lw=4, solid_capstyle="round", zorder=1)
        ax.scatter(i, obs_exact[i], color=dot_color, s=50, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("eigenvalue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2f}" for e in evals], rotation=45, fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="green",  label="exact inside"),
                       Patch(color="red",    label="exact outside"),
                       Patch(color="gold",   label="infeasible")], fontsize=8)

axes[0].set_ylabel("⟨Z₀⟩")

plt.tight_layout()
plt.savefig("new_cert_excited_states/bounds_comparison.pdf", bbox_inches="tight")
plt.show()
print("Plot saved to new_cert_excited_states/bounds_comparison.pdf")
