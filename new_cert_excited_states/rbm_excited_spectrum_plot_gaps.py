"""
Plot certification gap widths (ub - lb) per eigenstate: NPA2 baseline vs RBM-optimised.

Instead of showing the actual bound intervals, this script plots the gap width
directly as a grouped bar chart, making it easy to see which method is tighter
and by how much.

Loads the most recently modified rbm_excited_spectrum_*.npz from _outputs/,
or set TAG manually to select a specific run.

Run:
    cd scripts/python
    /path/to/venv/python3 -m new_cert_excited_states.rbm_excited_spectrum_plot_gaps
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Select result file
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_outputs"

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
    npz_path = candidates[-1]
else:
    npz_path = OUTPUT_DIR / f"rbm_excited_spectrum_{TAG}.npz"

meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
print(f"Loading: {npz_path}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

data      = np.load(npz_path)
evals     = data["evals"]
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
T_start  = meta.get("rbm_T_start", "?")
baseline_lbl = meta.get("baseline_label", "NPA2")
rbm_lbl      = meta.get("rbm_label", "RBM-opt")
total_w2 = meta.get("total_width_npa2")
total_wr = meta.get("total_width_rbm")

# ---------------------------------------------------------------------------
# Compute gaps
# ---------------------------------------------------------------------------

n = len(evals)

gaps_npa2 = np.array([
    float(ubs_npa2[i] - lbs_npa2[i])
    if not (np.isinf(lbs_npa2[i]) or np.isinf(ubs_npa2[i])) else np.nan
    for i in range(n)
])
gaps_rbm = np.array([
    float(ubs_rbm[i] - lbs_rbm[i])
    if not (np.isinf(lbs_rbm[i]) or np.isinf(ubs_rbm[i])) else np.nan
    for i in range(n)
])

improvement = gaps_npa2 - gaps_rbm   # positive = RBM is tighter

if total_w2 is not None and total_wr is not None:
    print(f"Total width — baseline: {total_w2:.4f}  |  RBM-opt: {total_wr:.4f}  "
          f"|  reduction: {total_w2 - total_wr:.4f} ({100*(total_w2-total_wr)/total_w2:.1f}%)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

x     = np.arange(n)
width = 0.38

COLOR_NPA2 = "steelblue"
COLOR_RBM  = "mediumpurple"

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})

# --- Top panel: grouped bars of gap widths ---
ax = axes[0]

bars1 = ax.bar(x - width/2, gaps_npa2, width, color=COLOR_NPA2, alpha=0.85,
               label=f"baseline ({baseline_lbl})")
bars2 = ax.bar(x + width/2, gaps_rbm,  width, color=COLOR_RBM,  alpha=0.85,
               label=f"RBM-opt ({rbm_lbl})")

ax.set_ylabel("certification gap  (ub − lb)", fontsize=11)
ax.set_title(
    f"XX-TFIM  N={N}, J={J}, h={h}, g={g},  δE={dE}\n"
    f"Certification gap per eigenstate — baseline vs RBM-opt\n"
    f"RBM: {steps} steps,  α={alpha},  decay={decay},  T₀={T_start},  seed={seed}  "
    f"|  total width: baseline={total_w2:.4f}, RBM={total_wr:.4f}",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# --- Bottom panel: improvement per eigenstate ---
ax2 = axes[1]

colors = np.where(improvement >= 0, COLOR_RBM, "tomato")
ax2.bar(x, improvement, color=colors, alpha=0.85)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("reduction\n(baseline − RBM)", fontsize=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

# x-axis labels shared
ax2.set_xticks(x)
ax2.set_xticklabels([f"{e:.3f}" for e in evals], rotation=45, fontsize=7, ha="right")
ax2.set_xlabel("eigenstate energy", fontsize=11)

plt.tight_layout()

out_pdf = npz_path.with_name(npz_path.stem + "_gaps.pdf")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"Plot saved → {out_pdf}")
