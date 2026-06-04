import sys

import matplotlib.pyplot as plt

from Moment import *
import matplotlib

matplotlib.use('Agg')
from scipy.optimize import curve_fit


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_simulations.py params.txt")
        sys.exit(1)

    param_file = sys.argv[1]

    params = {}
    with open(param_file, 'r') as f:
        exec(f.read(), {}, params)

    Ns = params.get('Ns', [5])
    Js = params.get('Js', [1])
    gs = params.get('gs', [0.3])
    hs = params.get('hs', [0.3])
    npa_levels = params.get('npa_levels', [1])
    indexes = params.get('indexes', [0])
    eps_list = params.get('eps_list', [0.1])
    npa_flags = params.get('npa_flag', [True])
    scalar_flags = params.get('scalar_flag', [True])

    prev_index = -1

    os.makedirs('Plots', exist_ok=True)

    total_runs = len(Ns) * len(Js) * len(gs)* len(hs) * len(indexes) * len(eps_list) * len(npa_levels) * len(npa_flags) * len(scalar_flags)
    current_run = 0

    N_prev = -1
    g_prev = -1000
    J_prev = -1000
    GSs = []

    for N, J, g, h, index, eps, npa_level, npa_flag, scalar_flag in product(Ns, Js, gs, hs, indexes, eps_list, npa_levels, npa_flags, scalar_flags):
        current_run += 1
        print(f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g} h={h}")

        if N != N_prev:
            # rand_op1, symbolic_op = random_pm1_hermitian((N//2 + 1, 0, 0))
            rand_op1 = np.array([[0, 1], [1, 0]], dtype=complex)
            symbolic_op = [(1.0, Monomial("X", (N // 2 + 1, 0, 0), 1.0))]
            N_prev = N
        # print(f"N = {N}, index = {index}")
        base_basis = []
        extra_basis = []
        base_basis = NPA(npa_level, N, None, True)
        # print(f"Basis size: {len(basis)}")
        if npa_flag:
            # extra_basis = generate_neighbour_monomials(base_basis, 1, pbc=True)
            # extra_basis = get_H2_monomials(N, J, g)
            # print(len(extra_basis))
            # print((extra_basis))
            # combined_basis = base_basis + extra_basis
            combined_basis = base_basis + generate_neighbour_monomials(base_basis, 1,
                                                                       pbc=True)

            basis = []
            seen = set()
            for b in combined_basis:
                if b not in seen:
                    seen.add(b)
                    basis.append(b)
        else:
            basis = base_basis

        basis = sort_basis(basis, N)

        # print(basis)
        print(f"Basis size: {len(basis)}")

        LB = GS_sdp_LB(N, get_symbolic_hamiltonian(N, J, g, h), basis, False).value
        print(f"GS lower bound: {LB}")




if __name__ == "__main__":
    main()

