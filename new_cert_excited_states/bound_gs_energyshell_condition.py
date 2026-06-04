"""
Ground-state energy lower bound: Wang-Acín vs PSD energy-shell.

For each N in [3,4,5,6], TFIM with periodic BC, compares two methods
for certifying a lower bound on the ground-state energy E_0:

  Method 1 — Wang-Acín:
      M >= 0,  <H> <= E_1  (scalar upper bound, first excited state energy)
      minimise <H>

  Method 2 — PSD energy shell:
      M >= 0,  G = (E_1 - H) rho >= 0  (PSD upper cut),  [rho, H] = 0
      minimise <H>

E_1 is obtained from exact diagonalisation.

Run:
    cd "/Users/cisco/Desktop/Claude SDP ETH"
    .venv/bin/python -m new_cert_excited_states.bound_gs_energyshell_condition
"""

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.spins_sdp import build_block_reps, compile_operator_linear_form
from spins.pauli_logic import I_PAULI

from new_cert_excited_states.energy_shell_sdp import (
    _collect_free_moments,
    _make_ext_label_index,
    _compile_energy_shell_block,
    _compile_commutation_constraints,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

J            = 1.0
h            = 0.5
g            = 0.3
BC           = "periodic"
SIZES        = [3, 4]
ANSATZ_LEVEL = 2     # which eigenstate to use as upper bound (1 = first excited, 2 = second, ...)
BASIS_LEVEL  = 1     # NPA level for the main moment matrix (1 = NPA1, 2 = NPA2, ...)
MAX_EXACT_N  = 12    # exact diag only up to this N; larger N reuse E_ansatz from N=MAX_EXACT_N
MOSEK_TOL    = 1e-8

MOSEK_PARAMS = {
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": MOSEK_TOL,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   MOSEK_TOL,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   MOSEK_TOL,
}

# ---------------------------------------------------------------------------
# Shared helper: build M >= 0 blocks
# ---------------------------------------------------------------------------

def _moment_matrix(basis, sym):
    """Return (reps, y, constraints) with M >= 0 and normalisation."""
    reps, _ = build_block_reps(basis, sym)
    rep0    = reps[0]
    m       = len(rep0.labels)
    y       = cp.Variable(m, name="y")
    cons    = [y[rep0.idx_I] == 1.0]

    for rep in reps:
        n     = rep.label_idx.shape[0]
        rows  = np.arange(n * n, dtype=np.int32)
        cols  = rep.label_idx.reshape(-1, order="F").astype(np.int32)
        dataA = rep.a_coef.reshape(-1, order="F").astype(float)
        CA    = sp.coo_matrix((dataA, (rows, cols)), shape=(n * n, m)).tocsr()
        A     = cp.reshape(cp.Constant(CA) @ y, (n, n), order="F")
        if rep.b_coef is None or rep.b_coef.size == 0:
            cons.append(A >> 0)
        else:
            dataB = rep.b_coef.reshape(-1, order="F").astype(float)
            CB    = sp.coo_matrix((dataB, (rows, cols)), shape=(n * n, m)).tocsr()
            B     = cp.reshape(cp.Constant(CB) @ y, (n, n), order="F")
            cons.append(cp.bmat([[A, -B], [B, A]]) >> 0)

    return reps, y, cons


def _run(prob):
    prob.solve(solver="MOSEK", mosek_params=MOSEK_PARAMS)
    if prob.status in ("infeasible", "infeasible_inaccurate"):
        return np.inf
    return float(prob.value)


# ---------------------------------------------------------------------------
# Method 1 — Wang-Acín: M >= 0  +  <H> <= E_1
# ---------------------------------------------------------------------------

def wang_acin(basis, H_dict, E_1, sym):
    reps, y, cons = _moment_matrix(basis, sym)
    c_H = compile_operator_linear_form(reps[0], H_dict, symmetry_manager=sym)
    cons.append(c_H @ y <= E_1)
    return _run(cp.Problem(cp.Minimize(c_H @ y), cons))


# ---------------------------------------------------------------------------
# Method 2 — PSD energy shell: M >= 0  +  G >= 0  +  [rho, H] = 0
# ---------------------------------------------------------------------------

def psd_energy_shell(basis, shell_basis, H_dict, E_1, sym):
    reps, y, cons    = _moment_matrix(basis, sym)
    rep0             = reps[0]
    main_label_index = rep0.label_index
    free_moments     = _collect_free_moments(shell_basis, H_dict, main_label_index, sym)
    ext_label_index  = _make_ext_label_index(main_label_index, free_moments)
    m_main           = len(main_label_index)
    m_free           = len(free_moments)
    m_ext            = m_main + m_free

    yz = cp.hstack([y, cp.Variable(m_free, name="z")]) if m_free > 0 else y

    # G = (E_1 - H) rho >= 0
    n_s        = len(shell_basis)
    CA_g, CB_g = _compile_energy_shell_block(
        shell_basis, H_dict, E_1, -1, ext_label_index, m_ext, sym
    )
    A_g = cp.reshape(cp.Constant(CA_g) @ yz, (n_s, n_s), order="F")
    if CB_g is None:
        cons.append(A_g >> 0)
    else:
        B_g = cp.reshape(cp.Constant(CB_g) @ yz, (n_s, n_s), order="F")
        cons.append(cp.bmat([[A_g, -B_g], [B_g, A_g]]) >> 0)

    # [rho, H] = 0  (broad commutation, complete constraints only)
    test_paulis = [w for w in ext_label_index if w != I_PAULI]
    for c_vec in _compile_commutation_constraints(
        test_paulis, H_dict, ext_label_index, m_ext, sym
    ):
        cons.append(c_vec @ yz == 0.0)

    c_H = compile_operator_linear_form(rep0, H_dict, symmetry_manager=sym)
    return _run(cp.Problem(cp.Minimize(c_H @ y), cons))


# ---------------------------------------------------------------------------
# Loop over system sizes
# ---------------------------------------------------------------------------

exact_sizes = []   # N values where exact diag was done
exact_gs    = []   # exact E_0 for those N
lbs_wa      = []
lbs_psd     = []
E_ansatz_fixed = None   # set at N = MAX_EXACT_N, reused for larger N

print(f"XX-TFIM J={J}, h={h}, g={g}, {BC} BC  —  NPA{BASIS_LEVEL} main, NPA1 shell, ansatz level {ANSATZ_LEVEL}")
print(f"Exact diag up to N={MAX_EXACT_N}; larger N reuse E_ansatz from N={MAX_EXACT_N}")
print(f"{'='*70}")
print(f"{'N':>3}  {'E_0':>12}  {'E_ansatz':>12}  {'lb Wang-Acin':>14}  {'lb PSD shell':>14}")
print("-" * 60)

for N in SIZES:
    sym        = SymmetryManager(N=N, use_real_basis=True)
    basis      = generate_npa_basis(N, k=BASIS_LEVEL).words
    shell_npa1 = generate_npa_basis(N, k=1).words
    H_dict     = xx_tfising_hamiltonian_dict(N, J=J, h=h, g=g, boundary=BC)

    if N <= MAX_EXACT_N:
        H_exact  = xx_tfising_hamiltonian_exact(N, J=J, h=h, g=g, boundary=BC)
        evals    = np.sort([float(e) for e in H_exact.eigenenergies()])
        E_0      = evals[0]
        E_ansatz = evals[ANSATZ_LEVEL]
        exact_sizes.append(N)
        exact_gs.append(E_0)
        if N == MAX_EXACT_N:
            E_ansatz_fixed = E_ansatz
        e0_str = f"{E_0:+.6f}"
    else:
        E_ansatz = E_ansatz_fixed
        e0_str   = "     N/A  "

    lb_wa  = wang_acin(basis, H_dict, E_ansatz, sym)
    lb_psd = psd_energy_shell(basis, shell_npa1, H_dict, E_ansatz, sym)

    lbs_wa.append(lb_wa)
    lbs_psd.append(lb_psd)

    print(f"N={N:>2}  E_0={e0_str}  E_ansatz={E_ansatz:+.6f}  lb_wa={lb_wa:+.6f}  lb_psd={lb_psd:+.6f}")

print("-" * 60)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(exact_sizes, exact_gs, "ko-",  label="Exact E₀ (N≤12)",              lw=2,   ms=8)
ax.plot(SIZES,       lbs_wa,   "b^--", label="Wang-Acín",                   lw=1.5, ms=7)
ax.plot(SIZES,       lbs_psd,  "gs--", label="PSD energy shell + [ρ,H]=0",  lw=1.5, ms=7)

ax.set_xlabel("N")
ax.set_ylabel("E₀ lower bound")
ax.set_title(f"XX-TFIM  J={J}, h={h}, g={g},  {BC} BC\n"
             f"NPA{BASIS_LEVEL} main + NPA1 shell,  E_ansatz = eigenstate {ANSATZ_LEVEL}  (fixed from N={MAX_EXACT_N} for larger N)")
ax.set_xticks(SIZES)
ax.legend()
plt.tight_layout()
plt.savefig("new_cert_excited_states/bound_gs_vs_N.pdf", bbox_inches="tight")
plt.show()
print("Plot saved to new_cert_excited_states/bound_gs_vs_N.pdf")
