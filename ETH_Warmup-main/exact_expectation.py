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

    for N, J, g, h, index, eps, npa_level, npa_flag, scalar_flag in product(Ns, Js, gs, hs, indexes, eps_list,
                                                                            npa_levels, npa_flags, scalar_flags):
        current_run += 1
        # index = 2**(N - 1) # for middle spectrum simulation
        # energy = -5
        print(
            f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g}, h={h}, index={index}, eps={eps:.0e}, NPA={npa_level}, scalar={scalar_flag} TI Block_diag")
        # print(f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g}, window center={energy}, eps={eps:.0e}, NPA={npa_level}")

        if N != N_prev or J != J_prev or g != g_prev:
            H = build_hamiltonian(N, J, 0, g, h, 0, 0, 0)
        #     J_prev = J
        #     g_prev = g

        if N != N_prev:
            # rand_op1, symbolic_op = random_pm1_hermitian((N//2 + 1, 0, 0))
            rand_op1 = np.array([[0, 1], [1, 0]], dtype=complex)
            symbolic_op = [(1.0, Monomial("X", (N // 2 + 1, 0, 0), 1.0))]
            N_prev = N
        # print(f"N = {N}, index = {index}")

        base_basis = NPA(npa_level, N, None, True)


        if npa_flag:
            extra_basis1 = generate_neighbour_monomials(base_basis, m=1, l=1, pbc=True)
            # print(extra_basis1)
            # extra_basis2 = generate_neighbour_monomials(base_basis, m =  1, l = 2, pbc=True)
            # combined_basis = base_basis + extra_basis1
            combined_basis = base_basis + extra_basis1
            # combined_basis = base_basis + extra_basis1 + extra_basis2

            basis = []
            seen = set()
            for b in combined_basis:
                if b not in seen:
                    seen.add(b)
                    basis.append(b)
        else:
            basis = base_basis

        basis = sort_basis(basis, N)


        print(f"Basis size: {len(basis)}")

        print(f"Getting eigenstuff...{N}")
        start = time.time()

        if N > 11:
            eigenvalues, eigenvectors = eigsh(H, k=index + 2, which='SA')
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")

        expectation_values = get_expectation_TI(N, N // 2, rand_op1, eigenvalues, eigenvectors, tol=1e-8,
                                                method='mixed')
        # GSs.append(eigenvalues[index]/N)
        print(f"True value: {expectation_values[index]}")
        print(f"GS: {eigenvalues[index]}")
        print(f"GS/N: {eigenvalues[index] / N}")
        # energy = eigenvalues[index]




if __name__ == "__main__":
    main()

