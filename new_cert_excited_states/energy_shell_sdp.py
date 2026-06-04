"""
Excited-state property certification via energy-shell SDP constraints.

Implements the same formulation as Tino's (collaborator's) code, but inside
our modular framework so the optimization tools (PT / RBM / BO) work unchanged.

Two key improvements over cert_excited_states/energy_shell_sdp.py
------------------------------------------------------------------
1. FREE SHELL MOMENTS
   Triple-product moments  w_i * w_j * P_γ  (shell_basis × shell_basis × H)
   that are NOT in the main basis y-vector are introduced as FREE variables z.
   They are constrained ONLY by the energy-shell PSD blocks F ≽ 0 and G ≽ 0,
   not by M ≽ 0 (which allows any quantum state).  Constraining them solely
   through the energy-shell blocks is strictly tighter for the excited-state
   problem, matching Tino's approach of letting M_H introduce extra variables.

2. BROAD COMMUTATION
   [ρ, H] = 0 is tested against EVERY non-identity moment in the extended
   variable vector (y + z), not just the shell_basis words.  More commutation
   equalities → tighter feasible set.

Optimization compatibility
--------------------------
solve_excited_sdp() has the same signature as before.  The optimization tools
select which moments go into the MAIN basis (y / M ≽ 0).  The free shell
moments z are added automatically and are invisible to the optimizer.

Public API
----------
bound_observable_excited(...)    bound <O> for states in an energy window
solve_excited_sdp(...)           single-sense wrapper for optimization tools
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from spins.pauli_logic import (
    PauliWord, Operator, I_PAULI,
    multiply_words,
)
from spins.symmetry import SymmetryManager
from spins.spins_sdp import (
    ObservableBoundResult,
    build_block_reps,
    compile_operator_linear_form,
)

# i^p for p in {0,1,2,3}
_I_POW: Tuple[complex, complex, complex, complex] = (1+0j, 1j, -1+0j, -1j)


# ---------------------------------------------------------------------------
# Step 1 — Identify free shell moments
# ---------------------------------------------------------------------------

def _collect_free_moments(
    shell_basis: List[PauliWord],
    hamiltonian: Operator,
    main_label_index: Dict[PauliWord, int],
    symmetry_manager: SymmetryManager,
) -> Dict[PauliWord, int]:
    """
    Find all canonical moments that appear in the energy-shell blocks

        F[i,j] = ⟨w_i w_j H⟩ - E_L ⟨w_i w_j⟩
        G[i,j] = E_U ⟨w_i w_j⟩ - ⟨w_i w_j H⟩

    but are NOT in main_label_index.  These become free variables z,
    constrained only by F ≽ 0, G ≽ 0, and [ρ, H] = 0.

    Returns: ordered dict  canonical PauliWord → index in z  (0-based).
    """
    free: Dict[PauliWord, int] = {}

    for w_i in shell_basis:
        for w_j in shell_basis:
            p_ij, w_ij = multiply_words(w_i, w_j)

            # The -E * ⟨w_ij⟩ term
            c_ij = symmetry_manager.canonicalize(w_ij)
            if c_ij is not None and c_ij not in main_label_index and c_ij not in free:
                free[c_ij] = len(free)

            # The h_γ * ⟨w_ij * P_γ⟩ triple-product terms
            for w_gamma in hamiltonian:
                _, w_ijg = multiply_words(w_ij, w_gamma)
                c_ijg = symmetry_manager.canonicalize(w_ijg)
                if (c_ijg is not None
                        and c_ijg not in main_label_index
                        and c_ijg not in free):
                    free[c_ijg] = len(free)

    return free


def _make_ext_label_index(
    main_label_index: Dict[PauliWord, int],
    free_moments: Dict[PauliWord, int],
) -> Dict[PauliWord, int]:
    """
    Merge main_label_index and free_moments into one extended index.
    Free moments are placed after the main moments:
        index k < m_main  →  y[k]  (main variable, constrained by M ≽ 0)
        index k >= m_main →  z[k - m_main]  (free variable)
    """
    m_main = len(main_label_index)
    ext = dict(main_label_index)
    for word, i in free_moments.items():
        ext[word] = m_main + i
    return ext


# ---------------------------------------------------------------------------
# Step 2 — Energy-shell block compiler (works with extended label index)
# ---------------------------------------------------------------------------

def _rb_adjust(c: complex, word: PauliWord, use_real_basis: bool) -> complex:
    """Multiply c by i^{-n_Y(word)} when using the real basis."""
    if not use_real_basis:
        return c
    n_y = (word.x_mask & word.z_mask).bit_count()
    return c * _I_POW[(-n_y) & 3]


def _compile_energy_shell_block(
    shell_basis: List[PauliWord],
    hamiltonian: Operator,
    E_bound: float,
    sign: int,
    ext_label_index: Dict[PauliWord, int],
    m_ext: int,
    symmetry_manager: SymmetryManager,
) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix]]:
    """
    Build sparse matrices CA, CB  (shape n_s² × m_ext) such that

        vec(F)[flat] = (CA @ yz).real + i * (CB @ yz).real

    where yz = [y; z] is the extended variable vector and

        F[i,j] = sign * ⟨w_i w_j (H - E_bound)⟩.

    sign = +1  →  F = (H - E_L) ρ  ≽ 0
    sign = -1  →  G = (E_U - H) ρ  ≽ 0
    """
    n = len(shell_basis)
    use_rb = symmetry_manager.use_real_basis

    rows_A, cols_A, data_A = [], [], []
    rows_B, cols_B, data_B = [], [], []

    def _accum(flat, k, c):
        if abs(c.real) > 1e-15:
            rows_A.append(flat); cols_A.append(k); data_A.append(c.real)
        if abs(c.imag) > 1e-15:
            rows_B.append(flat); cols_B.append(k); data_B.append(c.imag)

    for bi, w_i in enumerate(shell_basis):
        for bj, w_j in enumerate(shell_basis):
            flat = bi + bj * n
            p_ij, w_ij = multiply_words(w_i, w_j)
            phase_ij = _I_POW[p_ij]

            # -sign * E_bound * ⟨w_ij⟩
            c_ij = symmetry_manager.canonicalize(w_ij)
            if c_ij is not None and c_ij in ext_label_index:
                coeff = _rb_adjust(-sign * E_bound * phase_ij, c_ij, use_rb)
                _accum(flat, ext_label_index[c_ij], coeff)

            # sign * Σ_γ h_γ ⟨w_ij * P_γ⟩
            for w_gamma, h_gamma in hamiltonian.items():
                h = float(complex(h_gamma).real)
                if abs(h) < 1e-15:
                    continue
                p_ijg, w_ijg = multiply_words(w_ij, w_gamma)
                total_phase = (p_ij + p_ijg) & 3
                c_ijg = symmetry_manager.canonicalize(w_ijg)
                if c_ijg is None or c_ijg not in ext_label_index:
                    continue
                coeff = _rb_adjust(sign * h * _I_POW[total_phase], c_ijg, use_rb)
                _accum(flat, ext_label_index[c_ijg], coeff)

    CA = sp.coo_matrix((data_A, (rows_A, cols_A)), shape=(n * n, m_ext)).tocsr()
    CB = (
        sp.coo_matrix((data_B, (rows_B, cols_B)), shape=(n * n, m_ext)).tocsr()
        if rows_B else None
    )
    return CA, CB


# ---------------------------------------------------------------------------
# Step 3 — Commutation constraint compiler (extended variable vector)
# ---------------------------------------------------------------------------

def _compile_commutation_constraints(
    test_paulis: List[PauliWord],
    hamiltonian: Operator,
    ext_label_index: Dict[PauliWord, int],
    m_ext: int,
    symmetry_manager: SymmetryManager,
    tol: float = 1e-12,
) -> List[np.ndarray]:
    """
    Compile  tr(P_β [ρ, H]) = 0  as linear equalities on yz = [y; z].

    For each test Pauli P_β:
        Σ_γ h_γ (i^{p(γ,β)} - i^{p(β,γ)}) ⟨P_{β⊕γ}⟩ = 0

    A constraint is only added if ALL non-commuting products P_β P_γ land
    inside ext_label_index.  If any product is missing (truncated), the
    whole constraint is skipped — a partial equality is spurious and can
    make the SDP infeasible even when a valid solution exists.

    Calling this with all non-identity words in ext_label_index replicates
    Tino's approach of testing commutation for every moment variable.
    """
    use_rb = symmetry_manager.use_real_basis
    constraints = []

    for w_beta in test_paulis:
        coeff_re = np.zeros(m_ext, dtype=float)
        coeff_im = np.zeros(m_ext, dtype=float)
        valid = True

        for w_gamma, h_gamma in hamiltonian.items():
            h = float(complex(h_gamma).real)
            if abs(h) < 1e-15:
                continue

            p_bg, w_bg = multiply_words(w_beta, w_gamma)
            p_gb, _    = multiply_words(w_gamma, w_beta)

            factor = _I_POW[p_gb] - _I_POW[p_bg]
            if abs(factor) < 1e-12:
                continue   # P_β and P_γ commute — no contribution, safe to skip

            c_bg = symmetry_manager.canonicalize(w_bg)
            if c_bg is None or c_bg not in ext_label_index:
                valid = False   # truncation would make this equality spurious
                break

            eff = h * factor
            if use_rb:
                n_y = (c_bg.x_mask & c_bg.z_mask).bit_count()
                eff = eff * _I_POW[(-n_y) & 3]

            k = ext_label_index[c_bg]
            coeff_re[k] += eff.real
            coeff_im[k] += eff.imag

        if not valid:
            continue

        if np.any(np.abs(coeff_re) > tol):
            constraints.append(coeff_re)
        if np.any(np.abs(coeff_im) > tol):
            constraints.append(coeff_im)

    return constraints


# ---------------------------------------------------------------------------
# Step 4 — SDP builder
# ---------------------------------------------------------------------------

def _build_excited_state_problem(
    reps,
    objective_op: Operator,
    sense: str,
    symmetry_manager: SymmetryManager,
    shell_basis: List[PauliWord],
    hamiltonian: Operator,
    E_L: float,
    E_U: float,
    use_commutation: bool,
    free_moments: Dict[PauliWord, int],
    ext_label_index: Dict[PauliWord, int],
    scalar_flag: bool = False,
) -> Tuple[cp.Problem, cp.Expression]:
    """
    Build a single min/max excited-state SDP.

    Variables
    ---------
    y  (m_main,) : main moments from the chosen basis, constrained by M ≽ 0
    z  (m_free,) : free shell moments (triple products not in main basis),
                   constrained only by F ≽ 0, G ≽ 0, and [ρ,H] = 0

    Constraints (scalar_flag=False, default)
    ----------------------------------------
    y[idx_I] == 1                 normalisation
    M(y) ≽ 0                     main moment-matrix PSD (one block per symmetry sector)
    F(y,z) ≽ 0                   (H - E_L) ρ ≽ 0  lower energy shell  [PSD block]
    G(y,z) ≽ 0                   (E_U - H) ρ ≽ 0  upper energy shell  [PSD block]
    c_vec @ [y;z] == 0  ∀ c_vec  [ρ, H] = 0 for all moments in ext_label_index

    Constraints (scalar_flag=True, cheaper / looser)
    -------------------------------------------------
    y[idx_I] == 1                 normalisation
    M(y) ≽ 0                     main moment-matrix PSD
    |⟨wᵢ wⱼ (H−E)⟩| ≤ δE ∀ i,j  element-wise scalar bounds (eq. 12), no PSD block
    c_vec @ [y;z] == 0  ∀ c_vec  [ρ, H] = 0  (unchanged)
    """
    rep0   = reps[0]
    m_main = len(rep0.labels)
    m_free = len(free_moments)
    m_ext  = m_main + m_free

    y = cp.Variable(m_main, name="y")

    if m_free > 0:
        z  = cp.Variable(m_free, name="z")
        yz = cp.hstack([y, z])
    else:
        z  = None
        yz = y

    constraints = []

    # --- Normalisation ---
    constraints.append(y[rep0.idx_I] == 1.0)

    # --- Main moment-matrix M ≽ 0  (y only, unchanged from standard SDP) ---
    for rep in reps:
        n    = rep.label_idx.shape[0]
        rows = np.arange(n * n, dtype=np.int32)
        cols = rep.label_idx.reshape(-1, order="F").astype(np.int32)

        dataA = rep.a_coef.reshape(-1, order="F").astype(float)
        CA    = sp.coo_matrix((dataA, (rows, cols)), shape=(n * n, m_main)).tocsr()
        A     = cp.reshape(cp.Constant(CA) @ y, (n, n), order="F")

        if rep.b_coef is None or rep.b_coef.size == 0:
            constraints.append(A >> 0)
        else:
            dataB = rep.b_coef.reshape(-1, order="F").astype(float)
            CB    = sp.coo_matrix((dataB, (rows, cols)), shape=(n * n, m_main)).tocsr()
            B     = cp.reshape(cp.Constant(CB) @ y, (n, n), order="F")
            constraints.append(cp.bmat([[A, -B], [B, A]]) >> 0)

    # --- Energy-shell constraints (F, G)  ---
    if scalar_flag:
        # Cheap element-wise version (eq. 12): |⟨wᵢ wⱼ (H−E)⟩| ≤ δE per entry.
        # No PSD blocks — replaces F ≽ 0 and G ≽ 0 with scalar linear inequalities.
        E_center = (E_L + E_U) / 2
        delta_E  = (E_U - E_L) / 2
        CA_s, CB_s = _compile_energy_shell_block(
            shell_basis, hamiltonian, E_center, +1,
            ext_label_index, m_ext, symmetry_manager,
        )
        flat_re = cp.Constant(CA_s) @ yz
        constraints.append(flat_re >= -delta_E)
        constraints.append(flat_re <=  delta_E)
        if CB_s is not None:
            flat_im = cp.Constant(CB_s) @ yz
            constraints.append(flat_im >= -delta_E)
            constraints.append(flat_im <=  delta_E)
    else:
        # Full PSD blocks F ≽ 0, G ≽ 0  (tight, default)
        n_s = len(shell_basis)
        for sign, E_bound in [(+1, E_L), (-1, E_U)]:
            CA_s, CB_s = _compile_energy_shell_block(
                shell_basis, hamiltonian, E_bound, sign,
                ext_label_index, m_ext, symmetry_manager,
            )
            A_s = cp.reshape(cp.Constant(CA_s) @ yz, (n_s, n_s), order="F")
            if CB_s is None:
                constraints.append(A_s >> 0)
            else:
                B_s = cp.reshape(cp.Constant(CB_s) @ yz, (n_s, n_s), order="F")
                constraints.append(cp.bmat([[A_s, -B_s], [B_s, A_s]]) >> 0)

    # --- Commutation [ρ, H] = 0  —  test all non-identity moments in yz ---
    if use_commutation:
        test_paulis = [w for w in ext_label_index if w != I_PAULI]
        comm_vecs = _compile_commutation_constraints(
            test_paulis, hamiltonian, ext_label_index, m_ext, symmetry_manager,
        )
        for c_vec in comm_vecs:
            constraints.append(c_vec @ yz == 0.0)

    # --- Objective  (observable moments are always in the main basis y) ---
    c_obj    = compile_operator_linear_form(rep0, objective_op, symmetry_manager=symmetry_manager)
    obj_expr = c_obj @ y
    objective = cp.Minimize(obj_expr) if sense == "min" else cp.Maximize(obj_expr)

    return cp.Problem(objective, constraints), obj_expr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bound_observable_excited(
    basis:             List[PauliWord],
    hamiltonian:       Operator,
    observable:        Operator,
    E_center:          float,
    delta_E:           float,
    symmetry_manager:  SymmetryManager,
    shell_basis:       Optional[List[PauliWord]] = None,
    use_commutation:   bool  = True,
    scalar_flag:       bool  = False,
    mosek_tol:         float = 1e-6,
    scs_eps:           float = 1e-4,
    solver:            str   = "MOSEK",
    verbose:           bool  = False,
) -> ObservableBoundResult:
    """
    Certify bounds on <O> for eigenstates in the energy window [E_center ± delta_E].

    Solves two SDPs (min and max) with:
      - Main moment-matrix M ≽ 0  (main basis moments y)
      - Energy-shell blocks F ≽ 0 and G ≽ 0  (y + free shell moments z)
      - [ρ, H] = 0 for all moments in the extended variable vector

    Parameters
    ----------
    basis          : NPA basis for the main M matrix (subject to optimization).
    hamiltonian    : H = Σ h_γ P_γ.
    observable     : O = Σ o_α P_α.  Must be expressible in main basis.
    E_center       : Centre of target energy window.
    delta_E        : Half-width: window = [E_center - delta_E, E_center + delta_E].
    symmetry_manager : Symmetry settings.
    shell_basis    : Basis for F, G rows/columns.  Defaults to `basis`.
                     Triple-product moments not in `basis` become free variables.
    use_commutation : Add [ρ, H] = 0 equalities (default True).
    scalar_flag    : If True, replace F ≽ 0 / G ≽ 0 PSD blocks with element-wise
                     scalar bounds |⟨wᵢ wⱼ (H−E)⟩| ≤ δE (cheaper, looser).
    mosek_tol      : MOSEK primal/dual/gap tolerance.
    scs_eps        : SCS convergence tolerance (used when solver="SCS").
    solver         : "MOSEK" (default, accurate) or "SCS" (faster, approximate).
    verbose        : Print solver output.

    Returns
    -------
    ObservableBoundResult(lb, ub, energy_lb, energy_ub).
    If infeasible: lb = +inf, ub = -inf  (no eigenvalue in window, rigorous cert.).
    """
    if delta_E < 0:
        raise ValueError(f"delta_E must be >= 0, got {delta_E}")

    E_L = E_center - delta_E
    E_U = E_center + delta_E

    if shell_basis is None:
        shell_basis = basis

    reps, _          = build_block_reps(basis, symmetry_manager)
    main_label_index = reps[0].label_index

    free_moments     = _collect_free_moments(
        shell_basis, hamiltonian, main_label_index, symmetry_manager
    )
    ext_label_index  = _make_ext_label_index(main_label_index, free_moments)

    if solver == "SCS":
        solver_opts = {"eps": scs_eps, "max_iters": 10000}
    else:
        solver_opts = {
            "mosek_params": {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": mosek_tol,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   mosek_tol,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   mosek_tol,
            }
        }

    def _solve(sense: str) -> float:
        problem, _ = _build_excited_state_problem(
            reps, observable, sense, symmetry_manager,
            shell_basis, hamiltonian, E_L, E_U,
            use_commutation, free_moments, ext_label_index,
            scalar_flag=scalar_flag,
        )
        problem.solve(solver=solver, verbose=verbose, **solver_opts)
        if problem.status in ("infeasible", "infeasible_inaccurate"):
            return np.inf if sense == "min" else -np.inf
        return float(problem.value)

    lb = _solve("min")
    ub = _solve("max")

    return ObservableBoundResult(lb=lb, ub=ub, energy_lb=E_L, energy_ub=E_U)


def solve_excited_sdp(
    basis:            List[PauliWord],
    hamiltonian:      Operator,
    observable:       Operator,
    E_center:         float,
    delta_E:          float,
    sense:            str,
    symmetry_manager: SymmetryManager,
    shell_basis:      Optional[List[PauliWord]] = None,
    use_commutation:  bool  = True,
    scalar_flag:      bool  = False,
    lower_cut:        bool  = True,
    mosek_tol:        float = 1e-6,
    verbose:          bool  = False,
) -> float:
    """
    Solve a single (min or max) excited-state SDP and return the objective value.

    This is the entry point for the optimization tools (PT / RBM / BO).
    The optimizer selects which moments go into `basis` (the main M matrix);
    the free shell moments are added automatically and are invisible to it.

    Returns np.inf (sense='min') or -np.inf (sense='max') when infeasible.
    """
    if shell_basis is None:
        shell_basis = basis

    reps, _          = build_block_reps(basis, symmetry_manager)
    main_label_index = reps[0].label_index

    free_moments    = _collect_free_moments(
        shell_basis, hamiltonian, main_label_index, symmetry_manager
    )
    ext_label_index = _make_ext_label_index(main_label_index, free_moments)

    problem, _ = _build_excited_state_problem(
        reps, observable, sense, symmetry_manager,
        shell_basis, hamiltonian,
        E_center - delta_E, E_center + delta_E,
        use_commutation, free_moments, ext_label_index,
        scalar_flag=scalar_flag,
    )

    solver_opts = {
        "mosek_params": {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   mosek_tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   mosek_tol,
        }
    }
    problem.solve(solver="MOSEK", verbose=verbose, **solver_opts)

    if problem.status in ("infeasible", "infeasible_inaccurate"):
        return np.inf if sense == "min" else -np.inf

    return float(problem.value)
