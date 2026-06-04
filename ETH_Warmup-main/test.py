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
    npa_levels = params.get('npa_levels', [1])
    indexes = params.get('indexes', [0])
    eps_list = params.get('eps_list', [0.1])
    npa_flags = params.get('npa_flag', [False])
    scalar_flags = params.get('scalar_flag', [False])

    prev_index = -1

    os.makedirs('Plots', exist_ok=True)


    total_runs = len(Ns) * len(Js) * len(gs) * len(indexes) * len(eps_list) * len(npa_levels) * len(npa_flags) * len(scalar_flags)
    current_run = 0

    N_prev = -1
    g_prev = -1000
    J_prev = -1000
    GSs = []
    

    for N, J, g, index, eps, npa_level, npa_flag, scalar_flag in product(Ns, Js, gs, indexes, eps_list, npa_levels, npa_flags, scalar_flags):
        current_run += 1
        # index = 2**(N - 1) # for middle spectrum simulation
        # energy = -5
        print(f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g}, index={index}, eps={eps:.0e}, NPA={npa_level}, scalar={scalar_flag} TI Block_diag")
        # print(f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g}, window center={energy}, eps={eps:.0e}, NPA={npa_level}")

        if N != N_prev or J != J_prev or g != g_prev:
            H = build_hamiltonian(N, J, 0, g, 0, 0, 0, 0)
        #     J_prev = J
        #     g_prev = g

        if N != N_prev:
            # rand_op1, symbolic_op = random_pm1_hermitian((N//2 + 1, 0, 0))
            rand_op1 = np.array([[0, 1], [1, 0]], dtype=complex)
            symbolic_op = [(1.0, Monomial("X", (N//2 + 1, 0, 0), 1.0))]
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
            combined_basis = base_basis + generate_neighbour_monomials(base_basis, 1, pbc=True) #+ get_H2_monomials(N, J, g)


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

        # LB = GS_sdp_LB(N, get_symbolic_hamiltonian(N, J, g), basis, False).value
        # print(f"GS lower bound: {LB}")
        
        # eigenvalues = None
        # eigenvectors = None
        print(f"Getting eigenstuff...{N}")
        start = time.time()
        # eigenvalues, eigenvectors = load_eigen(f"eigenvalues{N}.npz", H)
        # eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        # eigenvalues, eigenvectors = eigsh(H, k=index + 1, which='SA')
        if N > 11:
            eigenvalues, eigenvectors = eigsh(H, k=index + 2, which='SA')
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")
        #
        # if index >= len(eigenvalues):
        #     print(f"Index skipped: Index {index} is out of bounds for N={N} (Max: {len(eigenvalues)-1})")
        #     print(f"Index skipped: Index {index} is out of bounds for N={N} (Max: {N//2 - 1})")
        #     continue
        # print(eigenvalues)
        # for i in range(len(eigenvalues)):
        #     if 0.55 >= eigenvalues[i] >= -0.55:
        #         index = i
        #         print(eigenvalues[i])
        #         print(index)
        #         break
        # expectation_values = None
        # energy = None
        # # expectation_values = get_expectation(N, N//2, rand_op1, eigenvalues, eigenvectors)
        expectation_values = get_expectation_TI(N, N//2, rand_op1, eigenvalues, eigenvectors,tol = 1e-8, method = 'mixed')
        # GSs.append(eigenvalues[index]/N)
        print(f"True value: {expectation_values[index]}")
        print(f"GS: {eigenvalues[index]}")
        print(f"GS/N: {eigenvalues[index]/N}")
        # energy = eigenvalues[index]


        # # min_sdp_density_mat = None
        # # max_sdp_density_mat = None
        # # print("Solving density matrix SDP min...")
        # # min_sdp_density_mat = density_matrix_sdp(N, H, energy, eps, N//2, rand_op1, True).value
        #
        # # print("Solving density matrix SDP max...")
        # # max_sdp_density_mat = density_matrix_sdp(N, H, energy, eps, N//2, rand_op1, False).value
        #
        # if scalar_flag:
        #     eps = eps / len(basis)
        #
        #
        # min_sdp_moment = None
        # max_sdp_moment = None
        # print("Solving moment SDP min...")
        # start = time.time()
        # min_sdp_moment = moment_sdp_block_diagonal_TI(N, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, True, scalar_flag = scalar_flag)
        # # min_sdp_moment = moment_sdp_block_diagonal_TI_H2(N, J, g, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, True, scalar_flag = scalar_flag)
        # # min_sdp_moment = moment_sdp(N, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, True)
        # end = time.time()
        # print(f"Elapsed time: {end - start:.4f} seconds")
        #
        # print("Solving moment SDP max...")
        # start = time.time()
        # max_sdp_moment = moment_sdp_block_diagonal_TI(N, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, False, scalar_flag = scalar_flag)
        # # max_sdp_moment = moment_sdp_block_diagonal_TI_H2(N, J, g, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, False, scalar_flag = scalar_flag)
        # # max_sdp_moment = moment_sdp(N, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, False)
        # end = time.time()
        # print(f"Elapsed time: {end - start:.4f} seconds")
        #
        #
        # min_delta = None
        # max_delta = None
        # min_delta = min_sdp_moment.value - expectation_values[index]
        # max_delta = max_sdp_moment.value - expectation_values[index]
        #
        # # if npa_flag:
        # #     print(f"Results for NPA level 1 + nearest neighbours")
        # # else:
        # #     print(f"Results for NPA level 1")
        #
        # # if npa_flag:
        # #     if scalar_flag:
        # #         print(f"Results for NPA level 1 + nearest neighbours and scalar shell constraint TI Block_diag")
        # #     else:
        # #         print(f"Results for NPA level 1 + nearest neighbours TI Block_diag")
        # # else:
        # #     if scalar_flag:
        # #         print(f"Results for NPA level 1 and scalar shell constraint TI Block_diag")
        # #     else:
        # #         print(f"Results for NPA level 1 TI Block_diag")
        #
        # if npa_flag:
        #     if scalar_flag:
        #         print(f"Results for NPA level 1 + H2 terms and scalar shell constraint TI Block_diag")
        #     else:
        #         print(f"Results for NPA level 1 + H2 terms TI Block_diag")
        # else:
        #     if scalar_flag:
        #         print(f"Results for NPA level 1 and scalar shell constraint TI Block_diag")
        #     else:
        #         print(f"Results for NPA level 1 TI Block_diag")
        #
        # # print(f"Delta min bound density matrix: {min_sdp_density_mat - expectation_values[index]:.6}")
        # # print(f"Delta max bound density matrix: {max_sdp_density_mat - expectation_values[index]:.6}")
        # print(f"Delta min bound moment matrix: {min_delta:.6}")
        # print(f"Delta max bound moment matrix: {max_delta:.6}")
        # # print(f"Min bound difference: {np.abs(min_sdp_density_mat - min_sdp_moment.value)}")
        # # print(f"Max bound difference: {np.abs(max_sdp_moment.value - max_sdp_density_mat)}")
        # # print(f"Density matrix bound width: {max_sdp_density_mat - min_sdp_density_mat:.6e}")
        # print(f"Moment matrix bound width: {max_sdp_moment.value - min_sdp_moment.value:.6e}")
        #
        #
        #
        # valid_bounds = True
        # for val in [ min_sdp_moment.value, max_sdp_moment.value]:
        #     if np.isnan(val) or np.isinf(val):
        #         valid_bounds = False
        #         break
        #
        # if not valid_bounds:
        #     print(f"  [!] Skipping plot for N={N}, eps={eps:.0e} due to invalid solver bounds (NaN or Inf).")
        #     continue
        #
        #
        # if npa_flag:
        #     if scalar_flag:
        #         plot_name = f'NPA Level {npa_level} and H2 terms and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, state {index}, eps={eps * len(basis):.2}'
        #         # plot_name = f'NPA Level {npa_level} and nearest neighbours terms and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
        #     else:
        #         plot_name = f'NPA Level {npa_level} and H2 terms TI Block_diag, N={N}, J={J}, g={g}, state {index}, eps={eps:.2}'
        #         # plot_name = f'NPA Level {npa_level} and nearest neighbours terms TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
        # else:
        #     if scalar_flag:
        #         plot_name = f'NPA Level {npa_level} and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, state {index}, eps={eps * len(basis):.2}'
        #         # plot_name = f'NPA Level {npa_level} and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
        #     else:
        #         plot_name = f'NPA Level {npa_level} TI Block_diag, N={N}, J={J}, g={g}, state {index}, eps={eps:.2}'
        #         # plot_name = f'NPA Level {npa_level} TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # fig.suptitle(plot_name, fontsize=18, fontweight='bold')
        #
        #
        # ax1.scatter(expectation_values, eigenvalues,
        #             color='orange', alpha=0.6, edgecolors='k', s=50, zorder=3, label='Exact Value')
        #
        # # rect1_dens = patches.Rectangle((min_sdp_density_mat, energy - eps),
        # #                              max_sdp_density_mat - min_sdp_density_mat, 2 * eps,
        # #                              linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, zorder=2, label='Density SDP Window')
        # # ax1.add_patch(rect1_dens)
        #
        # # ax1.scatter([min_sdp_density_mat, max_sdp_density_mat], [energy, energy],
        # #             color='red', marker='o', s=40, edgecolors='k', zorder=4, label='Density SDP (Min/Max)')
        #
        # rect1_mom = patches.Rectangle((min_sdp_moment.value, energy - eps),
        #                              max_sdp_moment.value - min_sdp_moment.value, 2 * eps,
        #                              linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.4, zorder=2, label='Moment SDP Window')
        # ax1.add_patch(rect1_mom)
        #
        # ax1.scatter([min_sdp_moment.value, max_sdp_moment.value], [energy, energy],
        #             color='blue', marker='o', s=40, edgecolors='k', zorder=5, label='Moment SDP (Min/Max)')
        #
        # ax1.scatter(expectation_values[index], eigenvalues[index],
        #     color='green', edgecolors='k', s=80, zorder=6, label='Target State')
        #
        # ax1.set_xlim(-1, 1)
        # ax1.set_xlabel(f'Expectation Value $\\langle X_{N//2 + 1} \\rangle$', fontsize=14)
        # ax1.set_ylabel('Energy $E$', fontsize=14)
        # ax1.set_title('Combined SDP Bounds Comparison (Full View)', fontsize=16)
        # ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
        # ax1.legend(loc='upper right')
        #
        #
        # ax2.scatter(expectation_values, eigenvalues,
        #             color='orange', alpha=0.6, edgecolors='k', s=100, zorder=3, label='Exact Value')
        #
        # ax2.scatter(expectation_values[index], eigenvalues[index],
        #     color='green', edgecolors='k', s=150, zorder=6, label='Target State')
        #
        # # rect2_dens = patches.Rectangle((min_sdp_density_mat, energy - eps),
        # #                              max_sdp_density_mat - min_sdp_density_mat, 2 * eps,
        # #                              linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, zorder=2, label='Density SDP Window')
        # # ax2.add_patch(rect2_dens)
        #
        # # ax2.scatter([min_sdp_density_mat, max_sdp_density_mat], [energy, energy],
        # #             color='red', marker='o', s=40, edgecolors='k', zorder=4, label='Density SDP (Min/Max)')
        #
        # rect2_mom = patches.Rectangle((min_sdp_moment.value, energy - eps),
        #                              max_sdp_moment.value - min_sdp_moment.value, 2 * eps,
        #                              linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.4, zorder=2, label='Moment SDP Window')
        # ax2.add_patch(rect2_mom)
        #
        # ax2.scatter([min_sdp_moment.value, max_sdp_moment.value], [energy, energy],
        #             color='blue', marker='o', s=40, edgecolors='k', zorder=5, label='Moment SDP (Min/Max)')
        #
        # # zoom_min = min(min_sdp_density_mat, min_sdp_moment.value)
        # # zoom_max = max(max_sdp_density_mat, max_sdp_moment.value)
        # zoom_min = min_sdp_moment.value
        # zoom_max = max_sdp_moment.value
        #
        # ax2.set_xlim(zoom_min - 0.05 * np.abs(zoom_min), zoom_max + 0.05 * np.abs(zoom_max))
        # ax2.set_ylim(energy - eps - 0.5 * eps, energy + eps + 0.5 * eps)
        #
        # ax2.set_xlabel(f'Expectation Value $\\langle X_{N//2 + 1} \\rangle$', fontsize=14)
        # ax2.set_title('Combined SDP Bounds Comparison (Zoomed View)', fontsize=16)
        #
        # ax2.minorticks_on()
        # ax2.grid(True, which='major', linestyle='--', alpha=0.8, zorder=0)
        # ax2.grid(True, which='minor', linestyle=':', alpha=0.7, zorder=0)
        # ax2.legend(loc='upper right')
        #
        # plt.tight_layout()
        #
        #
        # safe_name = plot_name.replace(':', '-').replace(',', '')
        # plt.savefig(f"Plots/{safe_name}.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)
    

    

    # # print(unsorted_monomials)
    # N = 3
    # base_basis = []
    # extra_basis = []
    # base_basis = NPA(1, N, None, True)
    # # print(f"Basis size: {len(base_basis)}")
    # extra_basis = generate_neighbour_monomials(base_basis, 1,True)
    # combined_basis = base_basis + extra_basis
    # basis = []
    # seen = set()
    # for b in combined_basis:
    #     if b not in seen:
    #         seen.add(b)
    #         basis.append(b)

    # basis = base_basis
    # print(basis)
    # print(f"Basis size: {len(basis)}")
    # print(apply_ti(basis, N))
    # print(f"Basis size TI: {len(apply_ti(basis, N))}")
    # basis = apply_ti(basis, N)
    # basis = sort_basis(basis, N)
    # print(basis)
    # M_expr, var_dict, symbolic_matrix, _ = create_moment_matrix_vectorized(basis, N)
    # print(len(var_dict))

    
    # symbolic_matrix_TI = []
    # for i in range(len(basis)):
    #     symbolic_matrix_TI.append(apply_ti(symbolic_matrix[i], N))

    # print_symbolic_matrix(symbolic_matrix)
    # size = len(symbolic_matrix)
    # all_safe = True
    
    # # Start at 1 to ignore the border. Step forward by N.
    # for row_i in range(1, size, N):
    #     row_f = min(row_i + N, size)
        
    #     for col_i in range(1, size, N):
    #         col_f = min(col_i + N, size)
            
    #         # Check the block!
    #         if is_block_circulant(symbolic_matrix, row_i, row_f, col_i, col_f):
    #             pass # Safe, do nothing
    #         else:
    #             print(f"WRONG: Block at rows {row_i}:{row_f}, cols {col_i}:{col_f} failed.")
    #             all_safe = False
                
    # if all_safe:
    #     print("SUCCESS: The entire matrix is circulant by blocks!")
    #     export_symbolic_matrix_to_csv(symbolic_matrix)
    
    # print_circulant_vectors(get_circulant_vectors(symbolic_matrix, N))

    # _, _, symbolic_matrix = create_block_diagonal_moment_matrix(basis, N)
    # print_block_diagonal_matrices(symbolic_matrix, N)

    # for site in range(N):
    #     val = get_expectation(N, site, rand_op1, eigenvalues, eigenvectors)[index]
    #     print(f"Site {site} expectation: {val}")


    # # ---------------------------------------------------------
    # # PLOT THE GS ENERGY SCALING AFTER THE LOOP FINISHES
    # # ---------------------------------------------------------
    #     if prev_index != index and N == Ns[-1]:
    #
    #         if len(Ns) != len(GSs):
    #             continue
    #
    #         prev_index = index
    #         gs_mean = np.mean(GSs)
    #         gs_variance = np.var(GSs)
    #
    #         print(f"Mean of normalized GS energies: {gs_mean}")
    #         print(f"Variance of normalized GS energies: {gs_variance}")
    #
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #
    #         # Plot the exact normalized GS energies
    #         ax.plot(Ns, GSs, marker='o', markersize=8, linestyle='-', linewidth=2,
    #                 color='forestgreen', markeredgecolor='k', label='Exact Normalized GS Energy')
    #
    #         ax.set_xlabel('System Size ($N$)', fontsize=14)
    #         ax.set_ylabel('Normalized GS Energy ($E_0 / N$)', fontsize=14)
    #
    #         # Assuming J and g are fixed to the first element for the title
    #         plot_title = f'Scaling of Ground State Energy (J={Js[0]}, g={gs[0]})'
    #         ax.set_title(plot_title, fontsize=16, fontweight='bold')
    #
    #         ax.grid(True, linestyle='--', alpha=0.7)
    #         ax.set_xticks(Ns)  # Force x-ticks to be exactly the integer system sizes
    #         ax.legend(fontsize=12)
    #
    #         plt.tight_layout()
    #
    #         # Save the plot
    #         scaling_plot_name = f'GS_Scaling_J_{Js[0]}_g_{gs[0]}_order1.png'
    #         plt.savefig(f"Plots/{scaling_plot_name}.png", dpi=300, bbox_inches='tight')
    #         plt.close(fig)
    #         print(f"Scaling plot saved to Plots/{scaling_plot_name}")
    #
    #         # GSs = []
    #
    # if len(Ns) > 3:
    #     print("\n--- Finite-Size Scaling Analysis ---")
    #
    #     # Convert lists to numpy arrays for math
    #     N_data = np.array(Ns)
    #     E_data = np.array(GSs)
    #
    #     # Define standard FSS fitting functions
    #     def power_law_fit(N, e_inf, A, c):
    #         return e_inf + A * np.power(N, -c)
    #
    #     def exponential_fit(N, e_inf, A, xi):
    #         return e_inf + A * np.exp(-N / xi)
    #
    #     # Fit the data
    #     try:
    #         # Initial guesses: [bulk_energy, amplitude, exponent/correlation_length]
    #         p0_pow = [E_data[-1], 1.0, 2.0]
    #         popt_pow, _ = curve_fit(power_law_fit, N_data, E_data, p0=p0_pow, maxfev=10000)
    #
    #         p0_exp = [E_data[-1], 1.0, 1.0]
    #         popt_exp, _ = curve_fit(exponential_fit, N_data, E_data, p0=p0_exp, maxfev=10000)
    #
    #         # Predict N=100
    #         N_target = 100
    #         E_100_pow = power_law_fit(N_target, *popt_pow)
    #         E_100_exp = exponential_fit(N_target, *popt_exp)
    #
    #         print(f"Thermodynamic Limit (N->inf) [Power Law]: {popt_pow[0]}")
    #         print(f"Predicted GS Energy at N=100 [Power Law]: {E_100_pow}\n")
    #
    #         print(f"Thermodynamic Limit (N->inf) [Exponential]: {popt_exp[0]}")
    #         print(f"Predicted GS Energy at N=100 [Exponential]: {E_100_exp}")
    #
    #         # Plot the FSS extrapolation
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #
    #         # Create smooth lines for the fit curves
    #         N_smooth = np.linspace(min(Ns), 105, 500)
    #
    #         ax.scatter(N_data, E_data, color='k', zorder=5, label='Exact Diagonalization Data')
    #         ax.plot(N_smooth, power_law_fit(N_smooth, *popt_pow), 'r--', alpha=0.8, label=f'Power-Law Fit ($N^{{-{popt_pow[2]:.2f}}}$)')
    #         ax.plot(N_smooth, exponential_fit(N_smooth, *popt_exp), 'b-.', alpha=0.8, label=f'Exponential Fit ($e^{{-N/{popt_exp[2]:.2f}}}$)')
    #
    #         # Mark the N=100 prediction
    #         ax.scatter([100], [E_100_pow], color='red', marker='*', s=150, zorder=6, label='N=100 Prediction')
    #
    #         ax.set_xlabel('System Size $N$', fontsize=14)
    #         ax.set_ylabel('Normalized GS Energy $e_0(N)$', fontsize=14)
    #         ax.set_title('Finite-Size Scaling Extrapolation', fontsize=16)
    #         ax.grid(True, linestyle=':', alpha=0.6)
    #         ax.legend()
    #
    #         plt.savefig("Plots/Finite_Size_Scaling.png", dpi=300, bbox_inches='tight')
    #         plt.close(fig)
    #         print("Extrapolation plot saved to Plots/Finite_Size_Scaling.png")
    #
    #     except Exception as e:
    #         print(f"Curve fitting failed: {e}")
    #
    # idx_2_9 = [i for i, n in enumerate(Ns) if 2 <= n <= 9]
    # idx_10_23 = [i for i, n in enumerate(Ns) if 10 <= n <= 23]
    #
    # energies_2_9 = [GSs[i]/i for i in idx_2_9]
    # energies_10_23 = [GSs[i]/i for i in idx_10_23]
    #
    # if energies_2_9:
    #     var_2_9 = np.var(energies_2_9)
    #     print(f"Variance N=2 to 9: {var_2_9}")
    #
    # if energies_10_23:
    #     var_10_23 = np.var(energies_10_23)
    #     print(f"Variance N=10 to 23: {var_10_23}")
    #
    # def exponential_fit(N, e_inf, A, xi):
    #     return e_inf + A * np.exp(-N / xi)
    #
    # N_data = np.array(Ns)
    # E_data = np.array(GSs)
    #
    # p0_exp = [E_data[-1], 1.0, 1.0]
    # popt, pcov = curve_fit(exponential_fit, N_data, E_data, p0=p0_exp, maxfev=10000)
    #
    # # N_target = 100
    # # predicted_E_100 = exponential_fit(N_target, *popt)
    #
    # # perr = np.sqrt(np.diag(pcov))
    # # max_error_E_100 = exponential_fit(N_target, popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2])
    # # eps_window = abs(max_error_E_100 - predicted_E_100) * 10
    #
    # # print(f"Predicted center for N=100: {predicted_E_100}")
    # # print(f"Suggested eps window for N=100: {eps_window}")
    #
    # # ---------------------------------------------------------
    # # EVOLUTION OF PREDICTED E_0 AND EPSILON
    # # ---------------------------------------------------------
    #
    # print("\n--- Evolution of Predicted Center and Epsilon Window ---")
    #
    # # Define the target range for your predictions
    # target_Ns = np.arange(10, 101, 1)
    #
    # predicted_centers = []
    # suggested_eps = []
    # eps_powers_of_10 = []
    # norms = []
    #
    # try:
    #     perr = np.sqrt(np.diag(pcov))
    #
    #     for N_target in target_Ns:
    #         norm_pred_E = exponential_fit(N_target, *popt)
    #         norms.append(norm_pred_E)
    #         actual_pred_E = norm_pred_E * N_target
    #         predicted_centers.append(actual_pred_E)
    #
    #         norm_max_error_E = exponential_fit(N_target, popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2])
    #         actual_max_error_E = norm_max_error_E * N_target
    #
    #         eps_val = abs(actual_max_error_E - actual_pred_E) * 10
    #         eps_val = max(eps_val, 1e-6)
    #         suggested_eps.append(eps_val)
    #
    #         # Calculate the next power of 10 strictly greater than eps_val
    #         # Add a tiny buffer (1e-14) to handle exact power-of-10 edge cases
    #         exponent = np.ceil(np.log10(eps_val + 1e-14))
    #         power_of_10_val = 10 ** exponent
    #         eps_powers_of_10.append(power_of_10_val)
    #
    #     with open("Plots/Window_Parameters_Evolution.txt", "w") as f:
    #         header = "N\tPredicted_Center\tSuggested_Eps\tNext_Power_of_10\n"
    #         f.write(header)
    #         print(header.strip())
    #
    #         for n, center, e_val, p10_val in zip(target_Ns, predicted_centers, suggested_eps, eps_powers_of_10):
    #             line = f"{n}\t{center:.6f}\t{e_val:.6e}\t{p10_val:.1e}"
    #             print(line)
    #             f.write(line + "\n")
    #
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    #     fig.suptitle('Evolution of SDP Window Parameters', fontsize=16, fontweight='bold')
    #
    #     ax1.plot(target_Ns, predicted_centers, 'b-', linewidth=2)
    #     ax1.set_ylabel('Predicted Target Energy', fontsize=14)
    #     ax1.grid(True, linestyle='--', alpha=0.7)
    #     ax1.set_title('Predicted Energy Center vs System Size')
    #
    #     ax2.plot(target_Ns, suggested_eps, 'r-', linewidth=2)
    #     ax2.set_xlabel('Target System Size', fontsize=14)
    #     ax2.set_ylabel('Suggested Window Size', fontsize=14)
    #     ax2.set_yscale('log')
    #     ax2.grid(True, linestyle='--', alpha=0.7)
    #     ax2.set_title('Required Epsilon Window vs System Size')
    #
    #     plt.tight_layout()
    #     plt.savefig("Plots/Window_Parameter_Evolution.png", dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print(norms)
    #     print(sum(norms)/len(norms))
    # except Exception as e:
    #     print(e)



    # Data extracted from the logs
    # N_vals = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    #
    # # Basis size is the same for min and max SDPs
    # basis_sizes = np.array([61, 73, 85, 97, 109, 121, 133, 145, 157, 169])
    #
    # # Taking the average elapsed time between the min and max SDPs for a smoother curve
    # time_min = np.array([2.6740, 5.1949, 9.4511, 17.2074, 27.9672, 44.4338, 70.8193, 108.7966, 160.8254, 247.5905])
    # time_max = np.array([2.6762, 5.3442, 9.5193, 17.7452, 28.1752, 44.8368, 70.7048, 109.9750, 168.6121, 271.4731])
    # avg_times = (time_min + time_max) / 2
    #
    # # Calculate theoretical O(N^5) scaling, anchored to the last empirical time point
    # scale_factor = avg_times[-1] / (N_vals[-1] ** 5)
    # theoretical_N5 = scale_factor * (N_vals ** 5)
    #
    # # Create the plot
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    # # Plot 1: Empirical Elapsed Time (Left Y-axis)
    # color1 = 'tab:red'
    # ax1.set_xlabel('Number of Particles ($N$)', fontsize=14)
    # ax1.set_ylabel('Average Elapsed Time (seconds)', color=color1, fontsize=14)
    # line1 = ax1.plot(N_vals, avg_times, marker='o', color=color1, linewidth=2, label='Empirical Elapsed Time')
    #
    # # Plot 1b: Theoretical N^5 Time (Left Y-axis)
    # color1b = 'darkred'
    # line1b = ax1.plot(N_vals, theoretical_N5, marker='', color=color1b, linewidth=2, linestyle=':',
    #                   label='Theoretical $O(N^5)$')
    #
    # ax1.tick_params(axis='y', labelcolor=color1)
    # ax1.grid(True, linestyle='--', alpha=0.6)
    #
    # # Create a twin axis sharing the same X-axis
    # ax2 = ax1.twinx()
    #
    # # Plot 2: Basis Size (Right Y-axis)
    # # color2 = 'tab:blue'
    # # ax2.set_ylabel('Basis Size', color=color2, fontsize=14)
    # # line2 = ax2.plot(N_vals, basis_sizes, marker='s', color=color2, linewidth=2, linestyle='--', label='Basis Size')
    # # ax2.tick_params(axis='y', labelcolor=color2)
    #
    # # Add a combined legend
    # lines = line1 + line1b #+ line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper left', fontsize=12)
    #
    # # Title and layout
    # plt.title('Growth Comparison: Basis Size vs. Elapsed Time ($O(N^5)$)', fontsize=16, fontweight='bold')
    # fig.tight_layout()
    #
    # # Save the plot
    # plt.savefig('growth_comparison.png', dpi=300, bbox_inches='tight')
    # print("Plot saved as 'growth_comparison.png'")

if __name__ == "__main__":
    main()


# Mean of normalized GS energies: -1.0226382531288427
# Variance of normalized GS energies: 8.351811020179296e-10