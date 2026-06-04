"""
Plot certified bounds on <Z_0> for all eigenstates: fixed NPA2 vs RBM-optimised.

Loads pre-computed results from _outputs/rbm_excited_spectrum_<tag>.npz.
Edit the TAG constant below to select which run to plot.

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.rbm_excited_spectrum_plot
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Select which result file to plot
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"

# Set TAG to match the compute script's output, or pass None to auto-select
# the most recent file.
TAG: str | None = None

if TAG is None:
    candidates = sorted(
        OUTPUT_DIR.glob("rbm_excited_spectrum_*.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No rbm_excited_spectrum_*.npz found in {OUTPUT_DIR}.\n"
            "Run rbm_excited_spectrum_compute.py first."
        )
    npz_path = candidates[-1]   # most recently modified
else:
    npz_path = OUTPUT_DIR / f"rbm_excited_spectrum_{TAG}.npz"

meta_path = npz_path.with_name(npz_path.stem + "_meta.json")

print(f"Loading: {npz_path}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

data      = np.load(npz_path)
evals     = data["evals"]
obs_exact = data["obs_exact"]
lbs_npa2  = data["lbs_npa2"]
ubs_npa2  = data["ubs_npa2"]
lbs_rbm   = data["lbs_rbm"]
ubs_rbm   = data["ubs_rbm"]

meta: dict = {}
if meta_path.exists():
    meta = json.loads(meta_path.read_text())

N        = meta.get("N", "?")
J        = meta.get("J", "?")
h        = meta.get("h", "?")
g        = meta.get("g", "?")
dE       = meta.get("delta_E", "?")
k        = meta.get("k", "?")
steps    = meta.get("rbm_steps", "?")
seed     = meta.get("rbm_seed", "?")
alpha    = meta.get("rbm_alpha", "?")
decay    = meta.get("rbm_decay", "?")
T_start  = meta.get("rbm_T_start", meta.get("T_start", "?"))
obs_lbl  = meta.get("observable", "Z₀")
total_w2 = meta.get("total_width_npa2")
total_wr = meta.get("total_width_rbm")

opt_per_state   = meta.get("optimize_per_eigenstate", None)
opt_target_idx  = meta.get("opt_eigenstate_index", None)  # None = all optimised independently

if opt_per_state is True:
    mode_str = "per-eigenstate optimisation"
elif opt_per_state is False:
    opt_e     = meta.get("opt_eigenstate_energy", None)
    opt_e_str = f"E={opt_e:+.4f}" if opt_e is not None else "?"
    mode_str  = f"middle-only (eigenstate {opt_target_idx}, {opt_e_str})"
else:
    mode_str = "?"

if total_w2 is not None and total_wr is not None:
    print(f"Total certified width — NPA2: {total_w2:.4f}  |  RBM-opt: {total_wr:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

x = np.arange(len(evals))

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g},  δE={dE}\n"
    f"Certified ⟨{obs_lbl}⟩ per eigenstate — NPA2 vs RBM-opt"
    f"  (k={k} from NPA3\\NPA1)\n"
    f"RBM: {steps} steps,  α={alpha},  decay={decay},  T₀={T_start},  seed={seed}"
    f"   |   {mode_str}",
    fontsize=10,
)

COLOR_RBM_DEFAULT = "mediumpurple"
COLOR_RBM_TARGET  = "darkorange"   # bar for the eigenstate the RBM was optimised on

for j in x:
    lb2, ub2 = float(lbs_npa2[j]), float(ubs_npa2[j])
    lb_r, ub_r = float(lbs_rbm[j]), float(ubs_rbm[j])

    # Green bar: NPA2 (thick, behind)
    if not (np.isinf(lb2) or np.isinf(ub2)):
        ax.plot([j, j], [lb2, ub2], color="green", lw=10,
                solid_capstyle="round", zorder=1, alpha=0.6)

    # RBM bar: orange if this is the optimisation target, purple otherwise
    if not (np.isinf(lb_r) or np.isinf(ub_r)):
        rbm_color = COLOR_RBM_TARGET if j == opt_target_idx else COLOR_RBM_DEFAULT
        ax.plot([j, j], [lb_r, ub_r], color=rbm_color, lw=5,
                solid_capstyle="round", zorder=2)

    # Exact value
    ax.scatter(j, obs_exact[j], color="black", s=30, zorder=4)

ax.set_xlabel("eigenstate index (sorted by energy)", fontsize=10)
ax.set_ylabel(f"⟨{obs_lbl}⟩", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.2f}" for e in evals], rotation=45, fontsize=7)

baseline_lbl = meta.get("baseline_label", "moment=NPA2, shell=NPA1")
rbm_lbl      = meta.get("rbm_label",      f"RBM-opt shell  (k={k} from NPA3\\NPA1)")

legend_handles = [
    Patch(color="green",           label=baseline_lbl, alpha=0.6),
    Patch(color=COLOR_RBM_DEFAULT, label=rbm_lbl),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                  markersize=6, label="exact ⟨Z₀⟩"),
]
if opt_target_idx is not None:
    legend_handles.insert(2, Patch(color=COLOR_RBM_TARGET,
                                   label="RBM-opt  (optimisation target)"))
ax.legend(handles=legend_handles, fontsize=9)

plt.tight_layout()

out_pdf = npz_path.with_suffix(".pdf")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"Plot saved to {out_pdf}")
