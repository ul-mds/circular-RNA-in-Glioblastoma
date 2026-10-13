"""
J. Supplementary figures (embedded in Additional file 1).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from common import *

OKABE = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", sky="#56B4E9", purple="#CC79A7", grey="#8A8A8A")
SSCALE = 1.15  # supplementary figures: 15% larger panels, text and numbers
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7 * SSCALE, "axes.labelsize": 7.5 * SSCALE, "axes.titlesize": 7.5 * SSCALE,
    "xtick.labelsize": 7 * SSCALE, "ytick.labelsize": 7 * SSCALE, "legend.fontsize": 6.5 * SSCALE,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})
MM = 1 / 25.4; W = 170 * MM
SF = os.path.join(OUT, "supp_figs"); os.makedirs(SF, exist_ok=True)
RES = f"{OUT}/results"

def lab(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10 * SSCALE, fontweight="bold", va="bottom")

# S1 imputation sensitivity
e1 = pd.read_csv(f"{RES}/E1_imputation_sensitivity.csv")
fig, axs = plt.subplots(1, 2, figsize=(W, 62 * MM * SSCALE)); fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.25, wspace=0.28)
for ax, layer, l in zip(axs, ["circRNA-mRNA", "circRNA-protein"], "ab"):
    d = e1[e1.layer == layer]; x = np.arange(len(d))
    ax.bar(x, d.n_edges, color="#D9D9D9", width=0.6, label="Edges"); ax.bar(x, d.overlap_original, color=OKABE["blue"], width=0.6, label="Shared with median imputation")
    for xi, n_, j in zip(x, d.n_edges, d.jaccard): ax.text(xi, n_ + 60, f"J = {j:.2f}", ha="center", fontsize=6.3 * SSCALE)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" (original)", "\n(reference)").replace("-case", "-\ncase").replace(" (MICE-like)", "\n(chained)") for m in d.method], fontsize=6.3 * SSCALE)
    ax.set_ylabel(f"{layer} edges"); ax.set_ylim(0, d.n_edges.max() * 1.15); lab(ax, l)
axs[0].legend(frameon=False, loc="lower left", fontsize=6.2 * SSCALE, bbox_to_anchor=(0, -0.38), ncol=2)
fig.savefig(f"{SF}/S1.png", dpi=300, facecolor="white"); plt.close(fig)

# S2 permutation null
nul = pd.read_csv(f"{RES}/B3_permutation_null_edge_counts.csv").iloc[:, 0].values
fig, ax = plt.subplots(figsize=(W * 0.55, 55 * MM * SSCALE)); fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.2)
ax.hist(nul, bins=np.arange(-0.5, nul.max() + 1.5, 1), color=OKABE["grey"]); ax.set_yscale("log"); ax.set_xlabel("Edges passing FDR < 0.05 and |rho| >= 0.3 after shuffling sample labels"); ax.set_ylabel("Permutations")
ax.text(0.97, 0.9, f"Observed: 5316 edges\nNull mean: {nul.mean():.1f}, maximum: {nul.max()}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8 * SSCALE)
fig.savefig(f"{SF}/S2.png", dpi=300, facecolor="white"); plt.close(fig)

# S3 Kaplan-Meier for further variables
D = load(); S = D["samples"]; C = D["circ"]; clin = D["clinical"].loc[S].copy()
clin["time"] = np.where(clin.PATH_DIAG_TO_DEATH_DAYS.notna(), clin.PATH_DIAG_TO_DEATH_DAYS, clin.PATH_DIAG_TO_LAST_CONTACT_DAYS); clin["event"] = clin.PATH_DIAG_TO_DEATH_DAYS.notna().astype(int)
ok = (clin.time.notna() & (clin.time > 0)).values
sig = pd.read_csv(f"{DERIVED}/Supp.20_Biomarker_signature_scores_biomarker_signature_scores.csv").set_index("sample_id")["signature_score"].reindex(S)
feats = {"circ_VCAN": np.log2(C["circ_chr5_83537007_83542268_VCAN"] + 1), "circ_HIPK3": np.log2(C["circ_chr11_33286413_33287511_HIPK3"] + 1), "Ten-circRNA score": sig, "Detected circRNAs": (C > 0).sum(1)}
fig, axs = plt.subplots(1, 4, figsize=(W, 55 * MM * SSCALE)); fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22, wspace=0.35)
for ax, (nm, v), l in zip(axs, feats.items(), "abcd"):
    v = pd.Series(np.asarray(v), index=S)[ok]; t = clin.time[ok]; e = clin.event[ok]; hi = v >= v.median()
    lr = logrank_test(t[hi], t[~hi], e[hi], e[~hi])
    for m, lb, c_ in [(hi, "High", OKABE["red"]), (~hi, "Low", OKABE["blue"])]:
        KaplanMeierFitter().fit(t[m] / 30.4, e[m], label=lb).plot_survival_function(ax=ax, ci_show=False, color=c_, lw=1.1 * SSCALE)
    ax.set_title(f"{nm}\nlog-rank P = {lr.p_value:.2f}", fontsize=6.8 * SSCALE); ax.set_xlabel("Months"); ax.set_ylabel("Survival" if l == "a" else ""); ax.legend(frameon=False, loc="lower left", fontsize=6 * SSCALE)
fig.savefig(f"{SF}/S3.png", dpi=300, facecolor="white"); plt.close(fig)

# S4 pathology sensitivity
h1 = pd.read_csv(f"{RES}/H1_pathology_sensitivity.csv")
print(h1)
print("supplementary figures done")
