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

    _, lower_bounds_Toni, upper_bounds_Toni, bound_widths_Toni = parse_sdp_logs("logs/SCALING results Toni g_1_01 UB_LB.log")
    _, lower_bounds_Toni_comm, upper_bounds_Toni_comm, bound_widths_Toni_comm = parse_sdp_logs("logs/SCALING results Toni commutator g_1_01 UB_LB.log")
    _, lower_bounds, upper_bounds, bound_widths = parse_sdp_logs("logs/SCALING results g_1_01 UB_LB.log")
    exp_values, _, _, _ = parse_sdp_logs("logs/exact_GS_X_expect_g1_01_5_10_20.log")

    # for i in range(21, 58):
    #     exp_values.append(exp_values[15])
    # print(len(exp_values))
    # print(len(lower_bounds))
    # print(len(upper_bounds))
    # print(len(bound_widths))
    # print(len(lower_bounds_Toni))
    # print(len(upper_bounds_Toni))

    plot_name = f'COMPARISON J={Js[0]}, g={gs[0]}, state {indexes[0]}' #, eps={eps_list[0]:.2}
    # plot_name = f'COMPARISON J={Js[0]}, g={gs[0]}, energy={0}, eps={eps_list[0]:.2}'

    # bounds plot
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(plot_name , fontsize=18, fontweight='bold')

    ax.scatter(Ns, upper_bounds_Toni, color='navy', marker='.',edgecolors='k' , s=75, label='upper/lower bound Toni', zorder=4, alpha=0.6)
    ax.scatter(Ns, lower_bounds_Toni, color='navy', marker='.',edgecolors='k' , s=75, zorder=4, alpha=0.6)
    ax.scatter(Ns, upper_bounds_Toni_comm, color='teal', marker='.',edgecolors='k' , s=75, label='upper/lower bound Toni with commutator', zorder=4, alpha=0.6)
    ax.scatter(Ns, lower_bounds_Toni_comm, color='teal', marker='.',edgecolors='k' , s=75, zorder=4, alpha=0.6)
    ax.scatter(Ns, upper_bounds, color='darkorange', marker='.',edgecolors='k' , s=75, label='upper/lower bound', zorder=4, alpha=0.6)
    ax.scatter(Ns, lower_bounds, color='darkorange', marker='.',edgecolors='k' , s=75, zorder=4, alpha=0.6)


    ax.scatter(Ns[:len(exp_values)], exp_values, color='red', marker='.',edgecolors='k' , s=200, label='actual expectation value', zorder=3, alpha=0.6)


    ax.set_xlabel('System Size $N$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\langle X \rangle$', fontsize=14, fontweight='bold', rotation=0, labelpad=20)


    ax.set_xticks(Ns)


    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)


    ax.legend(loc='upper left', frameon=False, fontsize=12)

    plt.tight_layout()


    plt.savefig(f"Plots/{plot_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # bound widths plot

    plot_name_widths = f'BOUND WIDTHS J={Js[0]}, g={gs[0]}, state {indexes[0]}' #, eps={eps_list[0]:.2}

    fig_w, ax_w = plt.subplots(figsize=(16, 8))
    fig_w.suptitle(plot_name_widths, fontsize=18, fontweight='bold')

    plot_Ns_widths = Ns[:len(bound_widths)]

    ax_w.plot(Ns[:len(bound_widths)], bound_widths, color='purple', linestyle='--', linewidth=1.5, zorder=2, alpha=0.5)
    ax_w.scatter(Ns[:len(bound_widths)], bound_widths, color='purple', marker='o', edgecolors='k', s=100, zorder=3, alpha=0.8)

    ax_w.set_xlabel('System Size $N$', fontsize=14, fontweight='bold')
    ax_w.set_ylabel('Bound Width', fontsize=14, fontweight='bold')

    ax_w.set_xticks(plot_Ns_widths)

    ax_w.grid(True, linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()

    plt.savefig(f"Plots/{plot_name_widths}.png", dpi=300, bbox_inches='tight')

    plt.close(fig_w)

if __name__ == "__main__":
    main()