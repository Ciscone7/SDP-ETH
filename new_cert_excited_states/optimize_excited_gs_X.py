"""
Compare three methods for bounding <X_0> on the ground state of the
XX transverse-field Ising model, vs system size N.

  Method 1 — NEW PSD       : NPA1 basis, PSD energy shell + broad [rho,H]=0
  Method 2 — NEW scalar    : NPA1 basis, scalar energy shell + broad [rho,H]=0
  Method 3 — NEW PSD + PT  : PT selects K moments from NPA2\\NPA1, PSD energy shell

No exact observable values are computed; only certified SDP bounds are plotted.

Run:
    cd "/Users/cisco/Desktop/Claude SDP ETH"
    .venv/bin/python -m new_cert_excited_states.optimize_excited_gs_X
"""

import numpy as np
import matplotlib.pyplot as plt

from spins.basis_builder import generate_npa_basis
from spins.models import xx_tfising_hamiltonian_exact, xx_tfising_hamiltonian_dict
from spins.symmetry import SymmetryManager
from spins.pauli_logic import local_pauli

from new_cert_excited_states.energy_shell_sdp import solve_excited_sdp
from optimize.montecarlo import parallel_tempering

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

J        = 1.0
h        = 0.0
g        = 0.3
BC       = "periodic"
SIZES    = [4, 6]
DELTA_E  = 0.15
K_OPT              = 5     # NPA2\NPA1 words selected (randomly for 1&2, PT for 3)
RANDOM_SEED        = 0     # seed for random word selection in methods 1 & 2
PT_CHAINS          = None  # None = auto (num CPUs)
PT_EPOCHS          = 2
PT_STEPS_PER_EPOCH = 5
PT_T_MIN           = 0.01
PT_T_MAX           = 2.0
PT_SEED            = 42
MOSEK_TOL          = 1e-8

# Observable: X at site 0
O_dict = {local_pauli(0, "x"): 1.0}

# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

def make_basis_sets(N):
    """Return (npa1_words, adding_words) where adding = NPA2 \\ NPA1."""
    npa1     = generate_npa_basis(N, k=1).words
    npa2     = generate_npa_basis(N, k=2).words
    npa1_set = set(npa1)
    adding   = [w for w in npa2 if w not in npa1_set]
    return npa1, adding


# ---------------------------------------------------------------------------
# PT objective wrapper
# ---------------------------------------------------------------------------

class ExcitedStateObjective:
    """
    Wraps solve_excited_sdp for use with parallel_tempering(obj_uses_indices=True).

    Always uses sense="min" and always returns -val, consistent with
    SpinOptimizationObjective. For the upper bound, pass the negated observable
    so that minimising -<-O> = minimising <O> gives the tightest ub.

    PT minimises the objective:
      lb case: val = lb,  return -lb  → PT maximises lb  (tighter lower bound)
      ub case: val = -ub, return +ub  → PT minimises ub  (tighter upper bound)
    """

    def __init__(self, starting_set, adding_set, H_dict, O_dict,
                 E_center, delta_E, sym, shell_basis):
        self.starting_set = starting_set
        self.adding_set   = adding_set
        self.H_dict       = H_dict
        self.O_dict       = O_dict
        self.E_center     = E_center
        self.delta_E      = delta_E
        self.sym          = sym
        self.shell_basis  = shell_basis

    def __call__(self, indices):
        chosen = [self.adding_set[int(i)] for i in indices]
        basis  = self.starting_set + chosen
        val    = solve_excited_sdp(
            basis, self.H_dict, self.O_dict,
            self.E_center, self.delta_E, "min", self.sym,
            shell_basis=self.shell_basis, scalar_flag=False,
            mosek_tol=MOSEK_TOL,
        )
        return -float(val)


# ---------------------------------------------------------------------------
# Main loop  (guard required for multiprocessing spawn on macOS)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lbs = {1: [], 2: [], 3: []}
    ubs = {1: [], 2: [], 3: []}

    print(f"XX-TFIM  J={J}, h={h}, g={g},  {BC} BC")
    print(f"Observable: X₀    δE={DELTA_E}    K_opt={K_OPT}")
    print(f"PT: chains={PT_CHAINS}, epochs={PT_EPOCHS}, steps/epoch={PT_STEPS_PER_EPOCH}, "
          f"T=[{PT_T_MIN},{PT_T_MAX}]")
    print(f"{'='*72}")
    print(f"{'N':>3}  {'lb_PSD':>9}  {'ub_PSD':>9}  "
          f"{'lb_sc':>9}  {'ub_sc':>9}  {'lb_opt':>9}  {'ub_opt':>9}")
    print("-" * 72)

    for N in SIZES:
        sym     = SymmetryManager(N=N, use_real_basis=True)
        H_dict  = xx_tfising_hamiltonian_dict(N, J=J, h=h, g=g, boundary=BC)
        H_exact = xx_tfising_hamiltonian_exact(N, J=J, h=h, g=g, boundary=BC)
        E_0     = float(min(H_exact.eigenenergies()))

        npa1, adding = make_basis_sets(N)
        shell        = npa1
        k_eff        = min(K_OPT, len(adding))

        # Random selection of k_eff words for methods 1 & 2 (same budget as method 3)
        rng          = np.random.default_rng(RANDOM_SEED)
        rand_indices = rng.choice(len(adding), size=k_eff, replace=False)
        basis_rand   = npa1 + [adding[i] for i in rand_indices]

        # --- Method 1: NEW PSD, NPA1 + random words ---
        lb1 = solve_excited_sdp(basis_rand, H_dict, O_dict, E_0, DELTA_E, "min", sym,
                                 shell_basis=shell, scalar_flag=False, mosek_tol=MOSEK_TOL)
        ub1 = solve_excited_sdp(basis_rand, H_dict, O_dict, E_0, DELTA_E, "max", sym,
                                 shell_basis=shell, scalar_flag=False, mosek_tol=MOSEK_TOL)

        # --- Method 2: NEW scalar, NPA1 + random words ---
        lb2 = solve_excited_sdp(basis_rand, H_dict, O_dict, E_0, DELTA_E, "min", sym,
                                 shell_basis=shell, scalar_flag=True, mosek_tol=MOSEK_TOL)
        ub2 = solve_excited_sdp(basis_rand, H_dict, O_dict, E_0, DELTA_E, "max", sym,
                                 shell_basis=shell, scalar_flag=True, mosek_tol=MOSEK_TOL)

        # --- Method 3: NEW PSD + PT optimizer ---
        L = len(adding)

        O_dict_neg = {w: -c for w, c in O_dict.items()}

        obj_lb = ExcitedStateObjective(npa1, adding, H_dict, O_dict,
                                       E_0, DELTA_E, sym, shell)
        res_lb = parallel_tempering(
            obj_lb, N=L, k=k_eff,
            num_chains=PT_CHAINS, num_epochs=PT_EPOCHS,
            steps_per_epoch=PT_STEPS_PER_EPOCH,
            T_min=PT_T_MIN, T_max=PT_T_MAX,
            seed=PT_SEED, verbose=False, obj_uses_indices=True,
        )
        lb3 = -float(res_lb["best"]["value"])

        obj_ub = ExcitedStateObjective(npa1, adding, H_dict, O_dict_neg,
                                       E_0, DELTA_E, sym, shell)
        res_ub = parallel_tempering(
            obj_ub, N=L, k=k_eff,
            num_chains=PT_CHAINS, num_epochs=PT_EPOCHS,
            steps_per_epoch=PT_STEPS_PER_EPOCH,
            T_min=PT_T_MIN, T_max=PT_T_MAX,
            seed=PT_SEED, verbose=False, obj_uses_indices=True,
        )
        ub3 = float(res_ub["best"]["value"])    # already ub: objective returns +ub directly

        lbs[1].append(lb1); ubs[1].append(ub1)
        lbs[2].append(lb2); ubs[2].append(ub2)
        lbs[3].append(lb3); ubs[3].append(ub3)

        print(f"N={N:>2}  {lb1:+9.5f}  {ub1:+9.5f}  "
              f"{lb2:+9.5f}  {ub2:+9.5f}  {lb3:+9.5f}  {ub3:+9.5f}")

    print("-" * 72)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    colors = {1: "royalblue", 2: "darkorange", 3: "forestgreen"}
    labels = {1: "NEW PSD", 2: "NEW scalar", 3: "NEW PSD + PT"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for m in [1, 2, 3]:
        c = colors[m]
        ax.plot(SIZES, lbs[m], "v--", color=c, label=f"{labels[m]}  lb",
                lw=1.2, ms=7, markerfacecolor="white", markeredgewidth=1.5)
        ax.plot(SIZES, ubs[m], "^-",  color=c, label=f"{labels[m]}  ub",
                lw=1.2, ms=7)

    ax.set_xlabel("N")
    ax.set_ylabel("⟨X₀⟩ bound")
    ax.set_title(
        f"XX-TFIM  J={J}, h={h}, g={g},  {BC} BC\n"
        f"δE={DELTA_E},  K_opt={K_OPT},  PT epochs={PT_EPOCHS}×{PT_STEPS_PER_EPOCH} steps"
    )
    ax.set_xticks(SIZES)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("new_cert_excited_states/optimize_excited_gs_X.pdf", bbox_inches="tight")
    plt.show()
    print("Plot saved to new_cert_excited_states/optimize_excited_gs_X.pdf")
