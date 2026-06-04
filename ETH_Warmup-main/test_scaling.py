import sys
from Moment import *
import matplotlib
matplotlib.use('Agg')




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

    
    os.makedirs('Plots', exist_ok=True)

    total_start = time.time()
    total_runs = len(Ns) * len(Js) * len(gs)* len(hs) * len(indexes) * len(eps_list) * len(npa_levels) * len(npa_flags) * len(scalar_flags)
    current_run = 0

    N_prev = -1
    g_prev = -1000
    J_prev = -1000
    i = 0

    LBs, UBs = parse_bounds_logs("logs/GS_bounds_g_1_01_h_0_5.log")



    for N, J, g, h, index, eps, npa_level, npa_flag, scalar_flag in product(Ns, Js, gs, hs, indexes, eps_list, npa_levels, npa_flags, scalar_flags):
        current_run += 1
        # index = 2**(N - 1) # for middle spectrum simulation
        # energy = -5

        energy = UBs[i] - (UBs[i] - LBs[i]) * 0.5
        eps = np.abs((UBs[i] - LBs[i]))
        # print(energy, eps)


        # print(f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g}, index={index}, eps={eps:.0e}, NPA={npa_level}, scalar={scalar_flag} TI Block_diag")
        print(f"\n[{current_run}/{total_runs}] Starting N={N}, J={J}, g={g}, h={h}, window center={energy}, eps={eps:.0e}, NPA={npa_level}")
        
        # if N != N_prev or J != J_prev or g != g_prev:
        #     H = build_hamiltonian(N, J, 0, g, 0, 0, 0, 0)
        #     H = build_hamiltonian(N, J, 0, g, h, 0, 0, 0)
        #     J_prev = J
        #     g_prev = g
            
        if N != N_prev:
            # rand_op1, symbolic_op = random_pm1_hermitian((N//2 + 1, 0, 0))
            rand_op1 = np.array([[0, 1], [1, 0]], dtype=complex)
            symbolic_op = [(1.0, Monomial("X", (N//2 + 1, 0, 0), 1.0))]
            N_prev = N
            i += 1
        
        base_basis = []
        extra_basis = []
        base_basis = NPA(npa_level, N, None, True)
        # print(f"Basis size: {len(basis)}")
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
        # print(basis)
        print(f"Basis size: {len(basis)}")


        eigenvalues = None
        eigenvectors = None
        # print(f"Getting eigenstuff...{N}")
        # start = time.time()
        # eigenvalues, eigenvectors = load_eigen(f"eigenvalues{N}.npz", H)
        # eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        # # eigenvalues, eigenvectors = eigsh(H, k=index + 1, which='SA')
        # if N > 11:
        #     eigenvalues, eigenvectors = eigsh(H, k=index + 1, which='SA')
        # else:
        #     eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        # end = time.time()
        # print(f"Elapsed time: {end - start:.4f} seconds")
        #
        # if index >= len(eigenvalues):
        #     print(f"Index skipped: Index {index} is out of bounds for N={N} (Max: {len(eigenvalues) - 1})")
        #     print(f"Index skipped: Index {index} is out of bounds for N={N} (Max: {N // 2 - 1})")
        #     continue

        # expectation_values = None
        # energy = None
        # expectation_values = get_expectation(N, N//2, rand_op1, eigenvalues, eigenvectors)
        # expectation_values = get_expectation_TI(N, N // 2, rand_op1, eigenvalues, eigenvectors, tol=1e-8, method='mixed')
        # print(f"True value: {expectation_values[index]}")
        # energy = eigenvalues[index]
        # energy = N * -1.022638 # g = 0.3
        # energy = -5.113788665585296 # g = 0.3
        # energy = N * -1.5017 # g = 1.3
        # energy = 0
        # energy = eigenvalues[index + 1] - (eigenvalues[index + 1] - eigenvalues[index]) * 0.5
        # eps = (eigenvalues[index + 1] - eigenvalues[index]) * 0.45



        # min_sdp_density_mat = None
        # max_sdp_density_mat = None
        # print("Solving density matrix SDP min...")
        # min_sdp_density_mat = density_matrix_sdp(N, H, energy, eps, N//2, rand_op1, True).value
        
        # print("Solving density matrix SDP max...")
        # max_sdp_density_mat = density_matrix_sdp(N, H, energy, eps, N//2, rand_op1, False).value
        
        # if scalar_flag:
        #     eps = eps / len(basis)


        min_sdp_moment = None
        max_sdp_moment = None
        print("Solving moment SDP min...")
        start = time.time()
        # min_sdp_moment = moment_sdp_block_diagonal_TI(N, get_symbolic_hamiltonian(N, J, g, h), basis, symbolic_op, energy, eps, True, scalar_flag = scalar_flag)
        min_sdp_moment = moment_sdp_TONI(N, get_symbolic_hamiltonian(N, J, g, h), basis, symbolic_op, energy, eps, True, scalar_flag=scalar_flag)

        # min_sdp_moment = moment_sdp(N, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, True)
        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")
        
        print("Solving moment SDP max...")
        start = time.time()
        # max_sdp_moment = moment_sdp_block_diagonal_TI(N, get_symbolic_hamiltonian(N, J, g, h), basis, symbolic_op, energy, eps, False, scalar_flag = scalar_flag)
        max_sdp_moment = moment_sdp_TONI(N, get_symbolic_hamiltonian(N, J, g, h), basis, symbolic_op, energy, eps, False, scalar_flag=scalar_flag)

        # max_sdp_moment = moment_sdp(N, get_symbolic_hamiltonian(N, J, g), basis, symbolic_op, energy, eps, False)
        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")
        

        # min_delta = None
        # max_delta = None
        # min_delta = min_sdp_moment.value - expectation_values[index]
        # max_delta = max_sdp_moment.value - expectation_values[index]
        
        # if npa_flag:
        #     print(f"Results for NPA level 1 + nearest neighbours") 
        # else:
        #     print(f"Results for NPA level 1")
        
        if npa_flag:
            if scalar_flag:
                print(f"SCALING TONI Results for NPA level 1 + nearest neighbours and scalar shell constraint TI Block_diag")
            else:
                print(f"SCALING Results for NPA level 1 + nearest neighbours TI Block_diag")
        else:
            if scalar_flag:
                print(f"SCALING Results for NPA level 1 and scalar shell constraint TI Block_diag")
            else:
                print(f"SCALING Results for NPA level 1 TI Block_diag")

        # print(f"Delta min bound density matrix: {min_sdp_density_mat - expectation_values[index]:.6}")
        # print(f"Delta max bound density matrix: {max_sdp_density_mat - expectation_values[index]:.6}")
        # print(f"Delta min bound moment matrix: {min_delta:.6}")
        # print(f"Delta max bound moment matrix: {max_delta:.6}")
        # print(f"Min bound difference: {np.abs(min_sdp_density_mat - min_sdp_moment.value)}")
        # print(f"Max bound difference: {np.abs(max_sdp_moment.value - max_sdp_density_mat)}")
        # print(f"Density matrix bound width: {max_sdp_density_mat - min_sdp_density_mat:.6e}")

        print(f"Min bound moment matrix: {min_sdp_moment.value}")
        print(f"Max bound moment matrix: {max_sdp_moment.value}")
        # print(f"True value:              {expectation_values[index]}")
        # print(f"Delta min bound moment matrix: {min_sdp_moment.value - expectation_values[index]:.6}")
        # print(f"Delta max bound moment matrix: {max_sdp_moment.value - expectation_values[index]:.6}")
        print(f"Moment matrix bound width: {max_sdp_moment.value - min_sdp_moment.value:.6e}")
        
        

        valid_bounds = True
        for val in [ min_sdp_moment.value, max_sdp_moment.value]:
            if np.isnan(val) or np.isinf(val):
                valid_bounds = False
                break
                
        if not valid_bounds:
            print(f"  [!] Skipping plot for N={N}, eps={eps:.0e} due to invalid solver bounds (NaN or Inf).")
            continue


        if npa_flag:
            if scalar_flag:
                # plot_name = f'SCALING NPA Level {npa_level} and nearest neighbours terms and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, state {index}, eps={eps * len(basis):.2}'
                plot_name = f'SCALING TONI NPA Level {npa_level} and nearest neighbours terms and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, h={h}, window center={energy}, eps={eps:.2}'
            else:
                plot_name = f'SCALING NPA Level {npa_level} and nearest neighbours terms TI Block_diag, N={N}, J={J}, g={g}, h={h}, state {index}, eps={eps:.2}'
                # plot_name = f'NPA Level {npa_level} and nearest neighbours terms TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
        else:
            if scalar_flag:
                plot_name = f'SCALING NPA Level {npa_level} and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, h={h}, state {index}, eps={eps * len(basis):.2}'
                # plot_name = f'NPA Level {npa_level} and scalar shell constraint TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
            else:
                plot_name = f'SCALING NPA Level {npa_level} TI Block_diag, N={N}, J={J}, g={g}, h={h}, state {index}, eps={eps:.2}'
                # plot_name = f'NPA Level {npa_level} TI Block_diag, N={N}, J={J}, g={g}, window center={energy}, eps={eps:.2}'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(plot_name, fontsize=18, fontweight='bold')
        
       
        # ax1.scatter(expectation_values, eigenvalues, 
        #             color='orange', alpha=0.6, edgecolors='k', s=50, zorder=3, label='Exact Value')
        
        # rect1_dens = patches.Rectangle((min_sdp_density_mat, energy - eps), 
        #                              max_sdp_density_mat - min_sdp_density_mat, 2 * eps, 
        #                              linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, zorder=2, label='Density SDP Window')
        # ax1.add_patch(rect1_dens)
        
        # ax1.scatter([min_sdp_density_mat, max_sdp_density_mat], [energy, energy], 
        #             color='red', marker='o', s=40, edgecolors='k', zorder=4, label='Density SDP (Min/Max)')
        
        rect1_mom = patches.Rectangle((min_sdp_moment.value, energy - eps), 
                                     max_sdp_moment.value - min_sdp_moment.value, 2 * eps, 
                                     linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.4, zorder=2, label='Moment SDP Window')
        ax1.add_patch(rect1_mom)
        # ax1.axhline(y=energy, color='blue', linestyle='-', linewidth=2, alpha=0.4, zorder=2, label='Moment SDP Window Level')

        # ax1.scatter([min_sdp_moment.value, max_sdp_moment.value], [energy, energy], 
        #             color='blue', marker='o', s=100, edgecolors='k', zorder=5, label='Moment SDP (Min/Max)')
        
        ax1.scatter([min_sdp_moment.value, max_sdp_moment.value], [energy, energy], 
                    color='blue', marker='o', s=40, edgecolors='k', zorder=5, label='Moment SDP (Min/Max)')
        
        # ax1.scatter(expectation_values[index], eigenvalues[index], 
        #     color='green', edgecolors='k', s=40, zorder=6, label='Target State')
        
        ax1.set_xlim(-1, 1)
        ax1.set_xlabel(f'Expectation Value $\\langle X_{N//2 + 1} \\rangle$', fontsize=14)
        ax1.set_ylabel('Energy $E$', fontsize=14)
        ax1.set_title('Combined SDP Bounds Comparison (Full View)', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax1.legend(loc='upper right')
        
        
        # ax2.scatter(expectation_values, eigenvalues, 
        #             color='orange', alpha=0.6, edgecolors='k', s=100, zorder=3, label='Exact Value')
        
        # ax2.scatter(expectation_values[index], eigenvalues[index], 
        #     color='green', edgecolors='k', s=150, zorder=6, label='Target State')
        
        # rect2_dens = patches.Rectangle((min_sdp_density_mat, energy - eps), 
        #                              max_sdp_density_mat - min_sdp_density_mat, 2 * eps, 
        #                              linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, zorder=2, label='Density SDP Window')
        # ax2.add_patch(rect2_dens)
        
        # ax2.scatter([min_sdp_density_mat, max_sdp_density_mat], [energy, energy], 
        #             color='red', marker='o', s=40, edgecolors='k', zorder=4, label='Density SDP (Min/Max)')
        
        rect2_mom = patches.Rectangle((min_sdp_moment.value, energy - eps), 
                                     max_sdp_moment.value - min_sdp_moment.value, 2 * eps, 
                                     linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.4, zorder=2, label='Moment SDP Window')
        ax2.add_patch(rect2_mom)
        
        ax2.scatter([min_sdp_moment.value, max_sdp_moment.value], [energy, energy], 
                    color='blue', marker='o', s=40, edgecolors='k', zorder=5, label='Moment SDP (Min/Max)')
        

        #target state zoom
        # zoom_min = min(min_sdp_density_mat, min_sdp_moment.value)
        # zoom_max = max(max_sdp_density_mat, max_sdp_moment.value)
        zoom_min = min_sdp_moment.value
        zoom_max = max_sdp_moment.value
        
        ax2.set_xlim(zoom_min - 0.05 * np.abs(zoom_min), zoom_max + 0.05 * np.abs(zoom_max))
        ax2.set_ylim(energy - eps - 0.5 * eps, energy + eps + 0.5 * eps)


        # # test gap zoom
        # x_min_bound = min(min_sdp_moment.value, expectation_values[index + 1])
        # x_max_bound = max(max_sdp_moment.value, expectation_values[index],  expectation_values[index + 1])
        # x_padding = 0.1 * abs(x_max_bound - x_min_bound) if x_max_bound != x_min_bound else 0.1

        # ax2.set_xlim(x_min_bound - x_padding, x_max_bound + x_padding)

        # y_min_bound = eigenvalues[index]
        # y_max_bound = eigenvalues[index + 1]
        # y_padding = 0.1 * abs(y_max_bound - y_min_bound)

        # ax2.set_ylim(y_min_bound - y_padding, y_max_bound + y_padding)
        
        ax2.set_xlabel(f'Expectation Value $\\langle X_{N//2 + 1} \\rangle$', fontsize=14)
        ax2.set_title('Combined SDP Bounds Comparison (Zoomed View)', fontsize=16)
        
        ax2.minorticks_on()
        ax2.grid(True, which='major', linestyle='--', alpha=0.8, zorder=0)
        ax2.grid(True, which='minor', linestyle=':', alpha=0.7, zorder=0)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        
        safe_name = plot_name.replace(':', '-').replace(',', '')
        plt.savefig(f"Plots/{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    total_end = time.time()
    print(f"Total elapsed time: {total_end - total_start:.4f} seconds")
    

if __name__ == "__main__":
    sys.stdout = Logger("SCALING results Toni g_1_01 h_0_5 UB_LB.log")
    main()