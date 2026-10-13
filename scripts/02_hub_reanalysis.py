"""
B. Reproduce the original correlation screen, audit the hub claim, and re-do the hub analysis robustly.
"""
import numpy as np, pandas as pd, networkx as nx
from scipy.stats import rankdata, spearmanr
from common import *

rng = np.random.default_rng(42)
D = load()
n = len(D["samples"])
C, P = D["circ"], D["prot"]
M = D["mrna"].loc[:, ~D["mrna"].columns.duplicated()]   # keep first occurrence of duplicated gene symbols
PTN = "circ_chr7_137251230_137254974_PTN"
VCAN = "circ_chr5_83537007_83542268_VCAN"
HIPK3 = "circ_chr11_33286413_33287511_HIPK3"

def screen(X, Y, nx_top, ny_top, label):
    """replicates original: top-variance features, all pairs, BH over all tests, FDR<.05 & |rho|>=.3"""
    vx = X.var().values; vy = Y.var().values
    ix = np.argsort(-vx, kind="stable")[:nx_top]; iy = np.argsort(-vy, kind="stable")[:ny_top]
    xs = X.columns[ix]; ys = Y.columns[iy]
    R = spearman_matrix(X.values[:, ix], Y.values[:, iy])
    p = rho_to_p(R, n)
    q = bh(p.ravel()).reshape(p.shape)
    sig = (q < .05) & (np.abs(R) >= .3)
    ii, jj = np.where(sig)
    df = pd.DataFrame({"feature1": xs[ii], "feature2": ys[jj], "correlation": R[ii, jj], "pvalue": p[ii, jj], "fdr": q[ii, jj]})
    print(f"[{label}] tests={R.size}, significant={len(df)}")
    return df, xs, ys, R, p, q

# ---- 1. reproduce original
sig_cm, xs, ys, R, p, q = screen(C, M, 200, 500, "circRNA-mRNA (orig design: top200 var circ x top500 var mRNA)")
sig_cp, *_ = screen(C, P, 200, 500, "circRNA-protein (orig design)")
orig = sig_cm
sig_cm.to_csv(f"{OUT}/results/B1_reproduced_circRNA_mRNA_edges.csv", index=False)
sig_cp.to_csv(f"{OUT}/results/B1_reproduced_circRNA_protein_edges.csv", index=False)

# ---- 2. audit of original hub construction
head = orig.head(200)   # rows are ordered by circRNA variance rank, so only the most variable circRNAs appear
G200 = nx.Graph(); G200.add_edges_from(zip(head.feature1, head.feature2))
print("\nHub network from a truncated edge table (first 200 rows of the screen output):")
print(head.feature1.value_counts().to_string())
print("nodes", G200.number_of_nodes(), "edges", G200.number_of_edges())
# true degree in the full set of significant edges
deg_full = orig.feature1.value_counts()
rank = {k: int(deg_full.index.get_loc(k)) + 1 for k in [PTN, VCAN, HIPK3]}
print("\nTrue degree in the full set of significant edges:")
print("circRNAs with >=1 edge:", len(deg_full))
for k in [PTN, VCAN, HIPK3]:
    print(f"  {k.split('_')[-1]:6s} degree={deg_full.get(k,0):4d} rank={rank[k]} / {len(deg_full)}")
print("Top 10 by degree:\n", deg_full.head(10).to_string())

# ---- 3. proper hub analysis on full graph
G = nx.Graph()
for a, b, c in zip(orig.feature1, orig.feature2, orig.correlation):
    G.add_edge(a, b, w=abs(c))
circs = [x for x in G.nodes if x.startswith("circ_")]
deg = dict(G.degree()); bet = nx.betweenness_centrality(G); clo = nx.closeness_centrality(G)
try:
    eig = nx.eigenvector_centrality(G, max_iter=5000)
except Exception:
    eig = {k: np.nan for k in G}
hub = pd.DataFrame({"circRNA": circs, "degree": [deg[c] for c in circs], "betweenness": [bet[c] for c in circs],
                    "closeness": [clo[c] for c in circs], "eigenvector": [eig[c] for c in circs]})
hub["detect_prev"] = [(C[c] > 0).mean() for c in circs]
hub["mean_expr"] = [C[c].mean() for c in circs]
# original hub score definition used in the manuscript script: .3 deg_c + .25 betw + .25 clos + .2 eig  (deg centrality=deg/(N-1))
N = G.number_of_nodes()
hub["hub_score"] = .3 * hub.degree / (N - 1) + .25 * hub.betweenness + .25 * hub.closeness + .2 * hub.eigenvector
hub = hub.sort_values("hub_score", ascending=False).reset_index(drop=True)
hub["rank_hub_score"] = hub.index + 1
hub["rank_degree"] = hub.degree.rank(ascending=False, method="min").astype(int)
print("\nFULL-network hub table (top 12 by hub score):")
print(hub.head(12).assign(circRNA=lambda d: d.circRNA.str.replace("circ_", "")).round(3).to_string(index=False))
for k in [PTN, VCAN, HIPK3]:
    row = hub[hub.circRNA == k].iloc[0]
    print(f"  {k.split('_')[-1]}: hub-score rank {int(row.rank_hub_score)}, degree rank {int(row.rank_degree)}, detect_prev={row.detect_prev:.2f}")
hub.to_csv(f"{OUT}/results/B2_full_network_hub_table.csv", index=False)

# degree vs detection prevalence / expression
rho1 = spearmanr(hub.degree, hub.detect_prev); rho2 = spearmanr(hub.degree, hub.mean_expr)
print(f"\nDegree vs detection prevalence: rho={rho1[0]:.2f} (p={rho1[1]:.1e});  degree vs mean expression: rho={rho2[0]:.2f}")

# ---- 4. permutation null: sample-shuffle mRNA matrix
nperm = 100
xs_v = C.values[:, np.argsort(-C.var().values, kind="stable")[:200]]; ys_v = M.values[:, np.argsort(-M.var().values, kind="stable")[:500]]
nullcounts = []
for b in range(nperm):
    perm = rng.permutation(n)
    Rb = spearman_matrix(xs_v, ys_v[perm])
    pb = rho_to_p(Rb, n); qb = bh(pb.ravel()).reshape(pb.shape)
    nullcounts.append(int(((qb < .05) & (np.abs(Rb) >= .3)).sum()))
nullnom = []
for b in range(20):
    perm = rng.permutation(n)
    Rb = spearman_matrix(xs_v, ys_v[perm]); pb = rho_to_p(Rb, n)
    nullnom.append(int(((pb < .05) & (np.abs(Rb) >= .3)).sum()))
print(f"\nPermutation null (sample-shuffled mRNA, {nperm} reps): edges passing FDR<.05&|rho|>=.3: mean={np.mean(nullcounts):.1f}, max={max(nullcounts)}  vs observed {len(sig_cm)}")
pd.Series(nullcounts).to_csv(f"{OUT}/results/B3_permutation_null_edge_counts.csv", index=False)

# ---- 5. bootstrap of the hub ranking (resample samples with replacement)
B = 300
edges_per = []
for b in range(B):
    idx = rng.integers(0, n, n)
    Rb = spearman_matrix(xs_v[idx], ys_v[idx])
    pb = rho_to_p(Rb, n); qb = bh(pb.ravel()).reshape(pb.shape)
    s = ((qb < .05) & (np.abs(Rb) >= .3)).sum(1)
    edges_per.append(s)
E = np.array(edges_per)   # B x 200 circRNAs
ranks = np.array([rankdata(-e, method="min") for e in E])
bs = pd.DataFrame({"circRNA": xs, "deg_obs": (np.abs(R) >= .3)[:, :].astype(int).sum(1) * 0})
obs_deg = ((q < .05) & (np.abs(R) >= .3)).sum(1)
bs = pd.DataFrame({"circRNA": xs, "deg_obs": obs_deg, "deg_boot_median": np.median(E, 0), "rank_obs": rankdata(-obs_deg, method="min"),
                   "rank_boot_median": np.median(ranks, 0), "rank_boot_p5": np.percentile(ranks, 5, axis=0), "rank_boot_p95": np.percentile(ranks, 95, axis=0),
                   "frac_boot_top1": (ranks == 1).mean(0), "frac_boot_top10": (ranks <= 10).mean(0)}).sort_values("rank_obs")
print("\nBootstrap (B=300) of degree-ranking inside the original 200x500 design:")
print(bs.head(8).assign(circRNA=lambda d: d.circRNA.str.replace("circ_", "")).round(2).to_string(index=False))
print(bs[bs.circRNA.isin([PTN, VCAN, HIPK3])].assign(circRNA=lambda d: d.circRNA.str.split("_").str[-1]).round(2).to_string(index=False))
bs.to_csv(f"{OUT}/results/B4_hub_bootstrap_rank.csv", index=False)

# ---- 6. correlations of circ_PTN with PTN gene
for lab, mat in [("mRNA", M), ("protein", P)]:
    col = "PTN" if lab == "mRNA" else "PTN|PTN"
    if col in mat.columns:
        r_, p_ = spearmanr(C[PTN], mat[col]); print(f"circ_PTN vs PTN {lab}: rho={r_:.3f}, p={p_:.2e}")
raw = D["mrna_raw"]["PTN"]; ok = raw.notna()
print("  raw (no imputation) n=", ok.sum(), spearmanr(C.loc[ok, PTN], raw[ok]))
rawp = D["prot_raw"]["PTN|PTN"]; ok = rawp.notna()
print("  PTN protein raw n=", ok.sum(), spearmanr(C.loc[ok, PTN], rawp[ok]))
print("circ_PTN detection prevalence:", (C[PTN] > 0).mean(), "; zeros:", (C[PTN] == 0).sum(), "of", n)
