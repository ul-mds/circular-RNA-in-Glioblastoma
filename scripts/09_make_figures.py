"""
I. Publication figures for the BMC Genomic Data manuscript.
   Output: TIFF (LZW, 300 dpi, RGB), width 170 mm (full page), max height 225 mm, fonts >= 6.5 pt at final size.
   Panels are composed into single files, panel letters are lower case bold, keys are inside the graphics,
   titles and legends are NOT inside the files (they are in the manuscript).
"""
import io, os, re, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from common import *

OKABE = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", grey="#8A8A8A", black="#000000")
SCALE = 1.15  # figures 2 to 7: panel height and text 15% larger for readability
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = f"{OUT}/results"
MM = 1 / 25.4
W = 170 * MM
os.makedirs(FIG, exist_ok=True)


def set_plot_style(scale):
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7 * scale, "axes.labelsize": 7.5 * scale, "axes.titlesize": 7.5 * scale,
        "xtick.labelsize": 7 * scale, "ytick.labelsize": 7 * scale, "legend.fontsize": 6.5 * scale,
        "axes.linewidth": 0.6 * scale, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5 * scale, "ytick.major.size": 2.5 * scale, "lines.linewidth": 1.0 * scale,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })


set_plot_style(1.0)

def save(fig, name):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, facecolor="white"); plt.close(fig)
    im = Image.open(buf).convert("RGB")
    im.save(f"{FIG}/{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    print(name, im.size, f"{im.size[0]/300*25.4:.0f} x {im.size[1]/300*25.4:.0f} mm", f"{os.path.getsize(f'{FIG}/{name}.tif')/1e6:.2f} MB")

def lab(ax, s, dx=-0.16, dy=1.06, scale=1.0):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10 * scale, fontweight="bold", va="bottom", ha="left")

gene = lambda c: c.split("_")[-1]
D = load(); S = D["samples"]; C = D["circ"]; M = D["mrna"].loc[:, ~D["mrna"].columns.duplicated()]; P = D["prot"]
depth = (C > 0).sum(1)
PTN = "circ_chr7_137251230_137254974_PTN"; VCAN = "circ_chr5_83537007_83542268_VCAN"; HIPK3 = "circ_chr11_33286413_33287511_HIPK3"
COL = {"PTN": OKABE["red"], "VCAN": OKABE["blue"], "HIPK3": OKABE["green"]}

# ------------------------------------------------------------------ Figure 1: study design flowchart (no result plots)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

F1S = 1.15  # Figure 1 canvas and text 15% larger for readability
FH = 94 * F1S
fig = plt.figure(figsize=(W, FH * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 170)
ax.set_ylim(0, FH)
ax.axis("off")
TITLE_Y = FH - 2.5
CONTENT_TOP = FH - 10.5


def fcbox(x, y, w, h, text, fc="#FFFFFF", ec="#444444", lw=0.9, fs=7.0, bold=False, align="center", style="round", zorder=5):
    fs *= F1S
    if style == "round":
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.4", fc=fc, ec=ec, lw=lw, zorder=zorder))
    else:
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=zorder))
    ax.text(x + (w / 2 if align == "center" else 2.2), y + h / 2, text, ha=align, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.22, color="#222222", zorder=zorder + 1)


def arr(x1, y1, x2, y2, color="#444444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8 * F1S, lw=0.9, color=color,
                                 shrinkA=0.5, shrinkB=0.5))


def panel_title(lbl, x, title):
    ax.text(x, TITLE_Y, lbl, fontsize=11 * F1S, fontweight="bold", ha="left", va="top")
    ax.text(x + 5.5, TITLE_Y, title, fontsize=7.2 * F1S, fontweight="bold", ha="left", va="top", color="#222222")


def vlink(x, y_upper_bottom, y_lower_top, color="#444444"):
    """Vertical arrow from the bottom edge of the upper box to the top edge of the lower box."""
    if y_lower_top < y_upper_bottom:
        arr(x, y_upper_bottom, x, y_lower_top, color=color)


def cross_link(x_out, y_out, x_in, y_in, color="#444444"):
    """Horizontal/diagonal link from panel b (right edge) to panel c (left edge)."""
    ax.add_patch(
        FancyArrowPatch(
            (x_out, y_out), (x_in, y_in), arrowstyle="-|>", mutation_scale=8 * F1S, lw=0.9, color=color,
            shrinkA=0, shrinkB=0, zorder=15,
        )
    )


def pipeline_arrow(x, y_bottom_upper, y_top_lower, color="#444444"):
    """Panel b: upper step bottom -> next step top (downward; head inside lower box)."""
    gap = y_bottom_upper - y_top_lower
    if gap < 0.35:
        return
    inset = min(0.28, gap * 0.08)
    y0 = y_bottom_upper - inset
    y1 = y_top_lower - inset
    ax.add_patch(
        FancyArrowPatch(
            (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=8 * F1S, lw=0.9, color=color,
            shrinkA=0, shrinkB=0, zorder=12,
        )
    )

panel_title("a", 2, "Data sources and cohort")
fcbox(2, CONTENT_TOP - 9, 52, 8.8, "CPTAC glioblastoma (gbm_cptac_2021)\ncBioPortal, secondary analysis", fc="#EAF3FA", bold=True, fs=6.6)
ya, ha = CONTENT_TOP - 20, 7.5
fcbox(15, ya, 26, ha, "99 tumours\nwith omics files", fc="#FFFFFF", ec=OKABE["grey"], fs=6.4)
yb = CONTENT_TOP - 31
fcbox(15, yb, 26, ha + 1.2, "98 matched tumours\n(all four layers)", fc="#FFFFFF", ec=OKABE["blue"], fs=6.4, bold=True)
vlink(28, ya, yb + ha + 1.2)
yc = CONTENT_TOP - 42
fcbox(6, yc, 44, 9.5, "Clinical data and somatic mutations\nage, sex, survival, IDH1 and others", fc="#F7FBFF", ec=OKABE["grey"], fs=6.3, align="left")
vlink(28, yb, yc + 9.5)
layers = [
    ("circRNA", "3,670 junctions", OKABE["blue"]),
    ("mRNA (FPKM)", "34,594 genes", OKABE["green"]),
    ("Protein (TMT)", "10,409 proteins", OKABE["orange"]),
    ("microRNA", "284 miRNAs", OKABE["purple"]),
]
for i, (nm, cnt, col) in enumerate(layers):
    y = yc - 11 - i * 9.0
    ax.add_patch(FancyBboxPatch((6, y), 44, 7.5, boxstyle="round,pad=0.25,rounding_size=1.0", fc="white", ec=col, lw=1.15))
    ax.text(8.5, y + 3.75, nm, fontsize=6.6 * F1S, fontweight="bold", va="center", color=col)
    ax.text(48.5, y + 3.75, cnt, fontsize=6.5 * F1S, va="center", ha="right", color="#333333")
arr(50, CONTENT_TOP - 58, 56, CONTENT_TOP - 58, color=OKABE["blue"])

panel_title("b", 58, "Analysis workflow")
pipeline = [
    ("Preprocessing", "Missing values, median imputation\nfeature filtering per layer"),
    ("Conventional screen", "Variance-filtered Spearman correlations\ncircRNA-mRNA and circRNA-protein edges"),
    ("Network and hubs", "Full network, truncated table check\nbootstrap ranks, unfiltered screen"),
    ("Clustering", "k-means, Ward, NMF, consensus\nRandom Forest and depth diagnostics"),
    ("Host gene and layers", "circRNA-host mRNA, partial correlations\ndepth and mRNA PC adjustment"),
    ("ceRNA checks", "circRNA-miRNA screen, seed sites\nTargetScan positive control"),
    ("Clinical and external", "Cox and Kaplan-Meier models\nGSE196695 junction replication"),
]
BX, BW, BH, BGAP = 58, 54, 8.8, 3.2
BAX = BX + BW / 2
BSTEP = BH + BGAP
y = CONTENT_TOP - 9
step_edges = []
for i, (head, body) in enumerate(pipeline):
    fcbox(
        BX, y, BW, BH, f"{head}\n{body}", fc="#F8F8F8" if i % 2 else "#FFFFFF", ec="#666666",
        fs=6.2, align="left", style="rect", zorder=6,
    )
    step_edges.append((y, y + BH))
    y -= BSTEP
sens_h = 7.2
sens_y = step_edges[-1][0] - BGAP - sens_h
fcbox(
    BX, sens_y, BW, sens_h,
    "Sensitivity analyses\nimputation, pathology review, early-death exclusion",
    fc="#FFF8E8", ec=OKABE["orange"], fs=6.0, align="left", style="rect", zorder=6,
)
for j in range(len(step_edges) - 1):
    pipeline_arrow(BAX, step_edges[j][0], step_edges[j + 1][1])
pipeline_arrow(BAX, step_edges[-1][0], sens_y + sens_h, color=OKABE["orange"])

panel_title("c", 118, "Validation and outputs")
CX, CW = 118, 48
CH, CSTEP = 9.0, 13.0
cy = CONTENT_TOP - 9
boxes_c = [
    ("GSE196695 (GEO)\n26 primary GBM, 3 recurrent GBM\n18 healthy brain controls", "#FDF1E3", "#CCAA66", 10.0),
    ("CIRI2 circRNA calls (GRCh37)\nliftover hg38 to hg19, junction match", "#FFFFFF", OKABE["orange"], CH),
    ("Replication and specificity\nprevalence, abundance, GBM vs brain", "#FFFFFF", OKABE["green"], CH),
    ("Shared outputs\nanalysis code, derived tables\nQC metrics and figure set", "#EEF7F2", OKABE["green"], 10.0),
    ("Intended use\nmethod checks and reproducibility\nfor circRNA multi-omic studies", "#FFFFFF", "#888888", CH),
]
c_centers = []
for i, (txt, fc, ec, h) in enumerate(boxes_c):
    fcbox(CX, cy, CW, h, txt, fc=fc, ec=ec, fs=6.2, align="left", style="rect", zorder=6)
    c_centers.append(cy + h / 2)
    cy_next = cy - CSTEP
    if i < len(boxes_c) - 1:
        nh = boxes_c[i + 1][3]
        vlink(CX + CW / 2, cy, cy_next + nh)
    cy = cy_next

B_OUT = BX + BW
b_center = lambda i: step_edges[i][0] + BH / 2
# Clustering -> shared outputs (code, tables, figures)
cross_link(B_OUT, b_center(3), CX, c_centers[3], color=OKABE["green"])
# Clinical and external (GSE196695 replication) -> external cohort in panel c
cross_link(B_OUT, b_center(6), CX, c_centers[0], color=OKABE["blue"])

ax.plot([56, 56], [5, CONTENT_TOP - 1], color="#CCCCCC", lw=0.8, ls="--", zorder=0)
ax.plot([116, 116], [5, CONTENT_TOP - 1], color="#CCCCCC", lw=0.8, ls="--", zorder=0)
save(fig, "Figure_1")

set_plot_style(SCALE)

# ------------------------------------------------------------------ Figure 2: hubs
hub = pd.read_csv(f"{RES}/B2_full_network_hub_table.csv"); hub["g"] = hub.circRNA.map(gene)
boot = pd.read_csv(f"{RES}/B4_hub_bootstrap_rank.csv"); boot["g"] = boot.circRNA.map(gene)
unb = pd.read_csv(f"{RES}/E2_unbiased_screen_degree.csv"); unb["g"] = unb.circRNA.map(gene)
fig, axs = plt.subplots(2, 2, figsize=(W, 128 * MM * SCALE)); fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.085, wspace=0.28, hspace=0.38)
ax = axs[0, 0]; h = hub.sort_values("degree", ascending=False).reset_index(drop=True)
ax.bar(np.arange(len(h)) + 1, h.degree, color="#C9C9C9", width=1.0)
for g_, c_ in COL.items():
    r = h[h.g == g_].index[0]; ax.bar(r + 1, h.degree[r], color=c_, width=1.6); rk = int(h.rank_degree[r])
    ax.annotate(f"{g_}\nrank {rk}", (r + 1, h.degree[r]), xytext=(r + 14, h.degree[r] + 38 + (0 if g_ == "PTN" else 12 if g_ == "VCAN" else 0)), fontsize=6.5, color=c_, arrowprops=dict(arrowstyle="-", lw=0.5, color=c_))
for i in range(2): ax.text(i + 1.5, h.degree[i] + 4, h.g[i], fontsize=6.3, ha="left")
ax.set_xlabel("circRNA rank by degree"); ax.set_ylabel("Number of significant edges"); lab(ax, "a", scale=SCALE)
ax = axs[0, 1]
ax.scatter(hub.detect_prev, hub.degree, s=6, color="#B5B5B5", lw=0)
for g_, c_ in COL.items():
    r = hub[hub.g == g_].iloc[0]; ax.scatter(r.detect_prev, r.degree, s=22, color=c_, zorder=3); ax.text(r.detect_prev - 0.02, r.degree + 6, g_, color=c_, fontsize=6.5, ha="right")
rho = spearmanr(hub.degree, hub.detect_prev)[0]
ax.text(0.03, 0.95, f"Spearman rho = {rho:.2f}", transform=ax.transAxes, fontsize=6.8, va="top")
ax.set_xlabel("Detection prevalence (fraction of 98 tumours)"); ax.set_ylabel("Number of significant edges"); lab(ax, "b", scale=SCALE)
ax = axs[1, 0]
sel = list(boot.sort_values("rank_obs").head(6).g) + ["PTN", "VCAN", "HIPK3"]
bb = boot[boot.g.isin(sel)].drop_duplicates("g").sort_values("rank_boot_median", ascending=False).reset_index(drop=True)
for i, r in bb.iterrows():
    c_ = COL.get(r.g, "#555555")
    ax.plot([r.rank_boot_p5, r.rank_boot_p95], [i, i], color=c_, lw=2.2, solid_capstyle="butt"); ax.scatter(r.rank_boot_median, i, color="white", edgecolor=c_, s=16, zorder=3, lw=1)
    ax.scatter(r.rank_obs, i, marker="|", color="black", s=40, zorder=4)
ax.set_yticks(range(len(bb))); ax.set_yticklabels(bb.g); ax.set_xlabel("Hub rank in 300 bootstrap resamples"); ax.set_xlim(0, 125)
ax.plot([], [], color="#555555", lw=2.2, label="5th to 95th percentile"); ax.scatter([], [], color="white", edgecolor="#555555", s=16, label="Median"); ax.scatter([], [], marker="|", color="black", s=40, label="Observed")
ax.legend(frameon=False, loc="upper right", handletextpad=0.4); lab(ax, "c", dx=-0.2, scale=SCALE)
ax = axs[1, 1]; u = unb.sort_values("degree", ascending=False).reset_index(drop=True)
ax.bar(np.arange(len(u)) + 1, u.degree, color="#C9C9C9", width=1.0)
for g_, c_ in COL.items():
    r = u[u.g == g_].index[0]; ax.bar(r + 1, u.degree[r], color=c_, width=6)
    ax.annotate(f"{g_}\nrank {r+1}", (r + 1, u.degree[r]), xytext=(r + 120, u.degree[r] + 1200 + (0 if g_ == "PTN" else 800 if g_ == "VCAN" else 1600)), fontsize=6.5, color=c_, arrowprops=dict(arrowstyle="-", lw=0.5, color=c_))
ax.set_xlabel("circRNA rank by degree (782 circRNAs, unfiltered screen)"); ax.set_ylabel("Number of significant edges"); lab(ax, "d", scale=SCALE)
save(fig, "Figure_2")

# ------------------------------------------------------------------ Figure 3: clustering
q1 = pd.read_csv(f"{RES}/C1b_labels_with_QC.csv").set_index("sample_id").loc[S]
prev = (C > 0).mean(); keep = prev[prev >= .2].index
Z = np.log2(C[keep] + 1); Z = (Z - Z.mean()) / Z.std()
pcs = PCA(n_components=2, random_state=0).fit(Z); sc = pcs.transform(Z)
km = KMeans(2, n_init=50, random_state=0).fit(Z); cl = km.labels_
rho_d = spearmanr(sc[:, 0], depth)[0]
dm = [depth[cl == k].median() for k in (0, 1)]
mw = mannwhitneyu(depth[cl == 0], depth[cl == 1]).pvalue
print(f"[Fig3] robust PCA: PC1 {100*pcs.explained_variance_ratio_[0]:.1f}% PC2 {100*pcs.explained_variance_ratio_[1]:.1f}%; PC1 vs depth rho={rho_d:.2f}; k=2 cluster depth medians {dm} sizes {np.bincount(cl)} MWU p={mw:.2e}")
fig, axs = plt.subplots(2, 2, figsize=(W, 124 * MM * SCALE)); fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.095, wspace=0.42, hspace=0.42)
ax = axs[0, 0]
im = ax.scatter(q1.PC1, q1.PC2, c=q1.n_detected, cmap="viridis", s=14, lw=0)
ax.set_xscale("symlog", linthresh=50); ax.set_yscale("symlog", linthresh=50)
mino = q1.index[q1.nmf == 1]
for s_ in mino:
    ax.scatter(q1.loc[s_, "PC1"], q1.loc[s_, "PC2"], facecolor="none", edgecolor=OKABE["red"], s=60, lw=1.0)
ax.text(0.97, 0.05, "Circles mark the three samples\nof the small NMF cluster", transform=ax.transAxes, fontsize=6.3, ha="right", color=OKABE["red"])
ax.set_xlabel("PC1 (conventional preprocessing, symlog)"); ax.set_ylabel("PC2 (symlog)"); lab(ax, "a", scale=SCALE)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Detected circRNAs", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6)
ax = axs[0, 1]
ax.scatter(sc[:, 0], depth, c=np.where(cl == 0, OKABE["blue"], OKABE["orange"]), s=14, lw=0)
ax.set_xlabel("PC1 (782 circRNAs, log2 and z-scored)"); ax.set_ylabel("Detected circRNAs per tumour")
ax.text(0.04, 0.95, f"Spearman rho = {abs(rho_d):.2f}", transform=ax.transAxes, fontsize=6.8, va="top"); ax.scatter([], [], color=OKABE["blue"], s=14, label="k-means cluster 1"); ax.scatter([], [], color=OKABE["orange"], s=14, label="k-means cluster 2")
ax.legend(frameon=False, loc="lower right", handletextpad=0.2); lab(ax, "b", scale=SCALE)
ax = axs[1, 0]; pac = pd.read_csv(f"{RES}/C3_consensus_PAC.csv")
ax.bar(pac.k, pac.PAC, color=OKABE["sky"], width=0.6); ax.set_ylim(0, 0.9)
for k_, v in zip(pac.k, pac.PAC): ax.text(k_, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.5)
ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Ambiguous clustering score (PAC)"); lab(ax, "c", scale=SCALE)
ax = axs[1, 1]; c4 = pd.read_csv(f"{RES}/C4_cluster_agreement_mRNA_protein.csv"); x = np.arange(len(c4)); w_ = 0.26
ax.bar(x - w_, c4.ARI_circ_vs_mRNA, w_, color=OKABE["blue"], label="circRNA vs mRNA"); ax.bar(x, c4.ARI_circ_vs_protein, w_, color=OKABE["orange"], label="circRNA vs protein"); ax.bar(x + w_, c4.ARI_mRNA_vs_protein, w_, color=OKABE["grey"], label="mRNA vs protein")
ax.set_xticks(x); ax.set_xticklabels(c4.k.astype(int)); ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Adjusted Rand index"); ax.legend(frameon=False, loc="upper right"); ax.set_ylim(0, 0.8); lab(ax, "d", scale=SCALE)
save(fig, "Figure_3")

# ------------------------------------------------------------------ Figure 4: host gene and adjustment
rng = np.random.default_rng(11)
host = pd.read_csv(f"{RES}/E4_circ_host_rho.csv")
genes = [g for g in M.columns]
rand = []
for c in host.circ:
    g = rng.choice(genes)
    while g == gene(c): g = rng.choice(genes)
    rand.append(spearmanr(C[c], M[g])[0])
rand = np.array(rand)
print(f"[Fig4] host rho median {host.rho_host.median():.2f}; random median {np.median(rand):.2f}; frac>=0.3 host {np.mean(host.rho_host>=.3):.2f} random {np.mean(rand>=.3):.3f}")
fig, axs = plt.subplots(2, 2, figsize=(W, 124 * MM * SCALE)); fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.09, wspace=0.3, hspace=0.42)
ax = axs[0, 0]; bins = np.linspace(-0.6, 1.0, 41)
ax.hist(rand, bins=bins, color=OKABE["grey"], alpha=0.85, label="Random circRNA-gene pairs")
ax.hist(host.rho_host.dropna(), bins=bins, color=OKABE["blue"], alpha=0.75, label="circRNA and its host gene")
ax.set_xlabel("Spearman rho"); ax.set_ylabel("Number of circRNAs"); ax.legend(frameon=False, loc="upper left"); lab(ax, "a", scale=SCALE)
ax = axs[0, 1]
xx = np.log2(C[PTN] + 1); yy = np.log2(M["PTN"] + 1)
ax.scatter(xx, yy, s=14, color=OKABE["red"], lw=0); ax.set_xlabel("circ_PTN, log2(count + 1)"); ax.set_ylabel("PTN mRNA, log2(FPKM + 1)")
ax.text(0.96, 0.05, f"Spearman rho = {spearmanr(C[PTN], M['PTN'])[0]:.2f}", transform=ax.transAxes, fontsize=6.8, va="bottom", ha="right"); lab(ax, "b", scale=SCALE)
ax = axs[1, 0]; e3 = pd.read_csv(f"{RES}/E3_depth_adjustment.csv")
vals = [5316] + list(e3.n_edges); keepv = [5316] + list(e3.retained_of_original)
names = ["No adjustment", "Adjusted for\ndetection depth", "Depth and\nmRNA PC1-2"]
ax.bar(range(3), vals, color="#D9D9D9", width=0.6, label="Edges after adjustment"); ax.bar(range(3), keepv, color=OKABE["blue"], width=0.6, label="Also in original set")
for i, (v, k_) in enumerate(zip(vals, keepv)): ax.text(i, v + 80, str(v), ha="center", fontsize=6.5)
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylabel("circRNA-mRNA edges"); ax.set_ylim(0, 6200); ax.legend(frameon=False, loc="upper right"); lab(ax, "c", scale=SCALE)
ax = axs[1, 1]
r_prot = spearmanr(C[PTN], P["PTN|PTN"])[0]; r_mp = spearmanr(M["PTN"], P["PTN|PTN"])[0]
from scipy.stats import rankdata
def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    return spearmanr(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))[0]
r_part = partial(C[PTN].values, P["PTN|PTN"].values, M["PTN"].values)
print(f"[Fig4] circ_PTN-PTN protein rho={r_prot:.3f}; partial|mRNA={r_part:.3f}; mRNA-protein rho={r_mp:.3f}")
bars = [r_prot, r_part, r_mp]; cols = [OKABE["orange"], OKABE["purple"], OKABE["green"]]
ax.bar(range(3), bars, color=cols, width=0.55)
for i, v in enumerate(bars): ax.text(i, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center", fontsize=6.8)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["circ_PTN vs\nPTN protein", "circ_PTN vs PTN\nprotein, given mRNA", "PTN mRNA vs\nPTN protein"]); ax.set_ylabel("Spearman rho"); ax.set_ylim(-0.15, 0.85); lab(ax, "d", scale=SCALE)
save(fig, "Figure_4")

# ------------------------------------------------------------------ Figure 5: ceRNA
d1 = pd.read_csv(f"{RES}/D1_circRNA_miRNA_pairs_with_sequence.csv"); d2 = pd.read_csv(f"{RES}/D2_seed_site_analysis.csv"); d3 = pd.read_csv(f"{RES}/D3_miRNA_target_positive_control.csv")
lbl = [f"circ_{gene(c)} and {m.replace('hsa-','')}" for c, m in zip(d1.circRNA, d1.miRNA)]
fig, axs = plt.subplots(1, 3, figsize=(W, 68 * MM * SCALE)); fig.subplots_adjust(left=0.22, right=0.99, top=0.88, bottom=0.16, wspace=0.48)
ax = axs[0]; y = np.arange(len(d1))[::-1]
ax.axvspan(-0.7, 0, color="#E8F1F8", lw=0); ax.axvline(0, color="black", lw=0.6)
ax.scatter(d1.rho_imputed, y, color=OKABE["blue"], s=18, label="Median imputed", zorder=3)
ax.scatter(d1.rho_complete_case, y, marker="D", facecolor="white", edgecolor=OKABE["red"], s=16, label="Observed values only", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=6.2); ax.set_xlim(-0.7, 0.7); ax.set_xlabel("Spearman rho")
ax.legend(frameon=False, loc="lower right", fontsize=6.0, handletextpad=0.3); lab(ax, "a", dx=-1.05, scale=SCALE)
ax.text(-0.68, 5.35, "Negative sign", fontsize=6, color=OKABE["blue"], va="bottom")
ax = axs[1]
ax.barh(y, d2.total_sites, color=OKABE["sky"], height=0.55)
for yi, ts, p_ in zip(y, d2.total_sites, d2.p_vs_shuffled_seq): ax.text(ts + 0.05, yi, f"P = {p_:.2f}", va="center", fontsize=6.2)
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 3.2); ax.set_xlabel("Seed sites in the circRNA sequence"); ax.set_xticks([0, 1, 2, 3]); lab(ax, "b", dx=-0.1, scale=SCALE)
ax = axs[2]
ax.hist(d3.delta.dropna(), bins=np.linspace(-0.2, 0.1, 31), color=OKABE["grey"]); ax.axvline(0, color="black", lw=0.6); ax.axvline(d3.delta.median(), color=OKABE["red"], lw=1.2)
ax.text(0.03, 0.95, f"Median {d3.delta.median():.3f}", transform=ax.transAxes, color=OKABE["red"], fontsize=6.5, va="top")
ax.set_xlabel("Target minus non-target rho"); ax.set_ylabel("Number of miRNAs"); lab(ax, "c", scale=SCALE)
save(fig, "Figure_5")

# ------------------------------------------------------------------ Figure 6: external data
G1 = pd.read_csv(f"{RES}/G1_external_matching_all_circRNAs.csv"); G3 = pd.read_csv(f"{RES}/G3_external_GBM_vs_brain_matched.csv"); G2 = pd.read_csv(f"{RES}/G2_external_focus_circRNAs.csv")
rows = []
for f in sorted(glob.glob(f"{OUT}/cache/ext/raw/*_circ.txt.gz")):
    nm = re.sub(r"^GSM\d+_|_circ\.txt\.gz$", "", f.split("/")[-1]); n_ = sum(1 for _ in pd.read_csv(f, sep="\t", usecols=["chr"]).chr)
    g_ = "Healthy brain" if nm.startswith("hb") else "Primary GBM" if re.match(r"P\d+$", nm) else "Recurrent GBM" if re.match(r"R\d+$", nm) else "Blood"
    rows.append((g_, n_))
qc = pd.DataFrame(rows, columns=["group", "n"])
fig, axs = plt.subplots(2, 2, figsize=(W, 128 * MM * SCALE)); fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.095, wspace=0.3, hspace=0.46)
ax = axs[0, 0]
bins = [(0, .05, "<5%"), (.05, .2, "5 to <20%"), (.2, .5, "20 to <50%"), (.5, 1.01, "50% or more")]
f1, f2 = [], []
for lo_, hi_, _ in bins:
    sub = G1[(G1.prevalence >= lo_) & (G1.prevalence < hi_)]; mm = sub[sub.matched]
    f1.append(100 * (mm.det_GBM > 0).sum() / len(sub)); f2.append(100 * (mm.det_GBM >= 8).sum() / len(sub))
x = np.arange(4); ax.bar(x - 0.19, f1, 0.36, color=OKABE["blue"], label="Found in at least 1 external tumour"); ax.bar(x + 0.19, f2, 0.36, color=OKABE["orange"], label="Found in 25% or more of external tumours")
for xi, a, b in zip(x, f1, f2): ax.text(xi - 0.19, a + 1, f"{a:.0f}", ha="center", fontsize=6.3); ax.text(xi + 0.19, b + 1, f"{b:.0f}", ha="center", fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels([b[2] for b in bins]); ax.set_xlabel("Detection prevalence in CPTAC"); ax.set_ylabel("circRNAs replicated (%)"); ax.set_ylim(0, 128); ax.set_yticks([0, 25, 50, 75, 100]); ax.legend(frameon=False, loc="upper left", fontsize=6.2); lab(ax, "a", scale=SCALE)
ax = axs[0, 1]
mm = G1[G1.matched & (G1.det_GBM > 0)]; cp = C[mm.circRNA].mean().values
ax.scatter(cp, mm.mean_jr_GBM, s=3, color=OKABE["blue"], alpha=0.45, lw=0); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Mean CPTAC expression"); ax.set_ylabel("Mean external junction reads"); ax.text(0.04, 0.95, f"Spearman rho = {spearmanr(cp, mm.mean_jr_GBM)[0]:.2f}", transform=ax.transAxes, fontsize=6.8, va="top"); lab(ax, "b", scale=SCALE)
ax = axs[1, 0]
G3 = G3.copy(); G3["nlp"] = -np.log10(G3.fdr.clip(lower=1e-300))
ax.scatter(G3.lfc, G3.nlp, s=3, color="#B5B5B5", lw=0)
focus = {"PTN": "circ_chr7_137251230_137254974_PTN", "VCAN": "circ_chr5_83537007_83542268_VCAN", "HIPK3": "circ_chr11_33286413_33287511_HIPK3", "FAT3": "circ_chr11_92352096_92355404_FAT3"}
offs = {"PTN": (0.3, 0, "left"), "VCAN": (0.3, 0, "left"), "HIPK3": (0.3, 0, "left"), "FAT3": (-0.3, 0, "right")}
for g_, c_ in focus.items():
    r = G3[G3.circRNA == c_]
    if len(r):
        r = r.iloc[0]; col = COL.get(g_, OKABE["purple"]); ax.scatter(r.lfc, r.nlp, s=22, color=col, zorder=3)
        ax.text(r.lfc + offs[g_][0], r.nlp + offs[g_][1], g_, fontsize=6.5, color=col, ha=offs[g_][2], va="center")
ax.axhline(-np.log10(0.05), color="black", lw=0.5, ls="--"); ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("log2 junction ratio, GBM over healthy brain\n(negative values: higher in healthy brain)"); ax.set_ylabel("-log10 FDR"); lab(ax, "c", scale=SCALE)

ax = axs[1, 1]
order = ["Primary GBM", "Recurrent GBM", "Healthy brain"]; data = [qc[qc.group == g_].n.values for g_ in order]
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.0))
for patch, c_ in zip(bp["boxes"], [OKABE["blue"], OKABE["sky"], OKABE["green"]]): patch.set_facecolor(c_); patch.set_alpha(0.8)
for i, d_ in enumerate(data): ax.scatter(np.random.default_rng(i).normal(i + 1, 0.06, len(d_)), d_, s=7, color="#333333", lw=0, zorder=3)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f"{o}\n(n={len(d)})" for o, d in zip(order, data)]); ax.set_ylabel("circRNAs detected per sample"); lab(ax, "d", scale=SCALE)
save(fig, "Figure_6")

# ------------------------------------------------------------------ Figure 7: survival
clin = D["clinical"].loc[S].copy()
clin["time"] = np.where(clin.PATH_DIAG_TO_DEATH_DAYS.notna(), clin.PATH_DIAG_TO_DEATH_DAYS, clin.PATH_DIAG_TO_LAST_CONTACT_DAYS)
clin["event"] = clin.PATH_DIAG_TO_DEATH_DAYS.notna().astype(int); ok = clin.time.notna() & (clin.time > 0)
x = np.log2(C[PTN] + 1)[ok.values]; t = clin.time[ok]; e = clin.event[ok]; hi = x >= x.median()
lr = logrank_test(t[hi], t[~hi], e[hi], e[~hi])
F1 = pd.read_csv(f"{RES}/F1_univariable_cox.csv"); F4 = pd.read_csv(f"{RES}/F4_genomewide_cox_circRNA.csv")
fig = plt.figure(figsize=(W, 70 * MM * SCALE)); gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.9, bottom=0.2, wspace=0.62, width_ratios=[1, 1.25, 1])
ax = fig.add_subplot(gs[0])
for m, labl, c_ in [(hi, "High circ_PTN", OKABE["red"]), (~hi, "Low circ_PTN", OKABE["blue"])]:
    KaplanMeierFitter().fit(t[m] / 30.4, e[m], label=f"{labl} (n = {int(m.sum())})").plot_survival_function(ax=ax, ci_show=False, color=c_, lw=1.2)
ax.set_xlabel("Months from diagnosis"); ax.set_ylabel("Overall survival"); ax.legend(frameon=False, loc="upper right", fontsize=6.2, bbox_to_anchor=(1.05, 1.02)); ax.text(0.04, 0.05, f"Log-rank P = {lr.p_value:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5); lab(ax, "a", dx=-0.28, scale=SCALE)
ax = fig.add_subplot(gs[1])
names = {"circ_PTN": "circ_PTN", "circ_VCAN": "circ_VCAN", "circ_HIPK3": "circ_HIPK3", "PTN_mRNA": "PTN mRNA", "PTN_protein": "PTN protein", "signature_score": "Ten-circRNA score", "circ_depth(#detected)": "Detected circRNAs", "age(per 10y)": "Age (per 10 years)", "male": "Male sex", "IDH1_mut": "IDH1 mutation"}
f = F1[F1.variable.isin(names)].copy(); f["lbl"] = f.variable.map(names); f = f.set_index("variable").loc[list(names)].reset_index()
yy = np.arange(len(f))[::-1]
for yi, r in zip(yy, f.itertuples()):
    c_ = OKABE["red"] if r.variable == "age(per 10y)" else "#444444"
    ax.plot([r.lo, r.hi], [yi, yi], color=c_, lw=1.2); ax.scatter(r.HR, yi, color=c_, s=16, zorder=3)
ax.axvline(1, color="black", lw=0.6); ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"]); ax.set_xlim(0.07, 5)
ax.set_yticks(yy); ax.set_yticklabels(f.lbl); ax.set_xlabel("Hazard ratio (95% CI)"); lab(ax, "b", dx=-0.55, scale=SCALE)
ax = fig.add_subplot(gs[2])
p = np.sort(F4.p.values); exp_ = (np.arange(1, len(p) + 1) - 0.5) / len(p)
ax.scatter(-np.log10(exp_), -np.log10(p), s=4, color=OKABE["blue"], lw=0); mx = max(-np.log10(p).max(), 3) + 0.2
ax.plot([0, mx], [0, mx], color="black", lw=0.6); ax.set_xlabel("Expected -log10 P"); ax.set_ylabel("Observed -log10 P"); lab(ax, "c", scale=SCALE)
ax.text(0.04, 0.95, "782 circRNAs\nlowest FDR 0.98", transform=ax.transAxes, fontsize=6.5, va="top")
save(fig, "Figure_7")
print("done")
