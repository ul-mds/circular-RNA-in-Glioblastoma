"""
E. Statistical robustness of the circRNA-mRNA / circRNA-protein associations
  1) imputation sensitivity: median (original) vs complete-case vs kNN vs iterative (MICE-like)
  2) zero-inflation: detection-prevalence filter, unbiased screen over ALL mRNAs
  3) technical-depth adjustment (partial Spearman)
  4) host-gene / circRNA-vs-protein vs mRNA claims
  5) mRNA-protein positive control, and checks of numbers quoted in the manuscript
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from common import *

rng = np.random.default_rng(5)
D = load(); S = D["samples"]; n = len(S)
C = D["circ"]
Mfull = D["mrna"]; Pfull = D["prot"]
PTN = "circ_chr7_137251230_137254974_PTN"

def design(X, Y, nx_top=200, ny_top=500):
    ix = np.argsort(-X.var().values, kind="stable")[:nx_top]; iy = np.argsort(-Y.var().values, kind="stable")[:ny_top]
    return ix, iy

def edges_from(R, n_eff, xn, yn):
    p = rho_to_p(R, n_eff); q = bh(p.ravel()).reshape(p.shape)
    ii, jj = np.where((q < .05) & (np.abs(R) >= .3))
    return pd.DataFrame({"circ": xn[ii], "feat": yn[jj], "rho": R[ii, jj], "fdr": q[ii, jj]}), q

def pairwise_complete(Xv, Yraw):
    """Spearman with pairwise deletion; Yraw may have NaN"""
    R = np.zeros((Xv.shape[1], Yraw.shape[1])); N = np.full(R.shape, n)
    hasnan = np.isnan(Yraw).any(0)
    if (~hasnan).any():
        R[:, ~hasnan] = spearman_matrix(Xv, Yraw[:, ~hasnan])
    for j in np.where(hasnan)[0]:
        ok = ~np.isnan(Yraw[:, j]); y = rankdata(Yraw[ok, j]); y = (y - y.mean()) / y.std()
        Xr = np.apply_along_axis(rankdata, 0, Xv[ok]); Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
        R[:, j] = Xr.T @ y / ok.sum(); N[:, j] = ok.sum()
    return R, N

summary = []
for lab, Ymat, Yraw_df in [("circRNA-mRNA", Mfull.loc[:, ~Mfull.columns.duplicated()], D["mrna_raw"].loc[:, ~D["mrna_raw"].columns.duplicated()]),
                           ("circRNA-protein", Pfull, D["prot_raw"])]:
    ix, iy = design(C, Ymat)
    xn = C.columns[ix].values; yn = Ymat.columns[iy].values
    Xv = C.values[:, ix]
    base_R = spearman_matrix(Xv, Ymat.values[:, iy]); base, _ = edges_from(base_R, n, xn, yn)
    raw = Yraw_df[yn]
    print(f"\n=== {lab}: design 200 x {len(yn)}; cells missing (pre-imputation) in the 500 selected features: {100*raw.isna().values.mean():.1f}%; features with any NaN: {int(raw.isna().any().sum())}")
    results = {"median (original)": base_R}
    # complete case
    Rcc, Ncc = pairwise_complete(Xv, raw.values)
    # use per-pair n for p-values
    pcc = np.vectorize(lambda r, k: rho_to_p(np.array([r]), int(k))[0])(Rcc, Ncc) if False else 2 * stats.t.sf(np.abs(Rcc * np.sqrt((Ncc - 2) / (1 - np.clip(Rcc, -.999999, .999999) ** 2))), Ncc - 2)
    qcc = bh(pcc.ravel()).reshape(pcc.shape)
    ii, jj = np.where((qcc < .05) & (np.abs(Rcc) >= .3))
    cc = pd.DataFrame({"circ": xn[ii], "feat": yn[jj], "rho": Rcc[ii, jj]})
    results["complete-case"] = Rcc
    # kNN & iterative
    sd = (raw - raw.mean()) / raw.std()
    knn = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(sd), columns=yn)
    Rk = spearman_matrix(Xv, knn.values); results["kNN (k=5)"] = Rk
    it = IterativeImputer(estimator=BayesianRidge(), max_iter=8, random_state=0, n_nearest_features=40, sample_posterior=False)
    mi = pd.DataFrame(it.fit_transform(sd), columns=yn)
    Rm = spearman_matrix(Xv, mi.values); results["iterative (MICE-like)"] = Rm
    bset = set(zip(base.circ, base.feat))
    for name, Rx in results.items():
        e, _ = edges_from(Rx, n, xn, yn) if name != "complete-case" else (cc, None)
        s = set(zip(e.circ, e.feat))
        iu = np.isfinite(Rx) & np.isfinite(base_R)
        r_agree = spearmanr(Rx[iu], base_R[iu])[0]
        ptn_edges = int((e.circ == PTN).sum())
        print(f"   {name:24s} significant edges={len(e):5d}  overlap with original={len(s & bset):5d}  Jaccard={len(s & bset)/max(1,len(s | bset)):.2f}  rho-vs-original rho={r_agree:.3f}  circ_PTN edges={ptn_edges}")
        summary.append(dict(layer=lab, method=name, n_edges=len(e), overlap_original=len(s & bset), jaccard=len(s & bset) / max(1, len(s | bset)),
                            rho_agreement=r_agree, circ_PTN_edges=ptn_edges))
pd.DataFrame(summary).to_csv(f"{OUT}/results/E1_imputation_sensitivity.csv", index=False)

# ---------- 2. zero inflation & unbiased screen ----------
print("\n=== Zero inflation / detection prevalence")
prev = (C > 0).mean()
ix0, _ = design(C, Mfull)
sel_prev = prev.values[ix0]
print(f"  Of the 200 circRNAs selected by variance for the original screen: detection prevalence <20% for {(sel_prev<.2).sum()}, <50% for {(sel_prev<.5).sum()}; median {np.median(sel_prev):.2f}")
print(f"  (of all 3,670 circRNAs: median prevalence {prev.median():.3f}; 782 have >=20%)")
keep = np.where(prev.values >= .2)[0]
Mdup = Mfull.loc[:, ~Mfull.columns.duplicated()]
R = spearman_matrix(C.values[:, keep], Mdup.values)
p = rho_to_p(R, n)
# BH over ~27M tests (vectorised)
flat = p.ravel(); o = np.argsort(flat); m = flat.size
q = np.empty(m); qs = flat[o] * m / (np.arange(m) + 1); qs = np.minimum.accumulate(qs[::-1])[::-1]; q[o] = np.minimum(qs, 1)
q = q.reshape(p.shape)
sig = (q < .05) & (np.abs(R) >= .3)
deg = sig.sum(1); cn = C.columns[keep]
print(f"  Unbiased screen: {len(keep)} circRNAs (prevalence>=20%) x {Mdup.shape[1]} mRNAs = {R.size/1e6:.1f}M tests; significant edges = {sig.sum()}; circRNAs with >=1 edge = {(deg>0).sum()}")
dg = pd.Series(deg, index=cn).sort_values(ascending=False)
print("  Top 10 circRNAs by degree in unbiased screen:", [(i.replace('circ_', ''), int(v)) for i, v in dg.head(10).items()])
print(f"  circ_PTN degree = {int(dg.get(PTN, 0))}; rank = {int(list(dg.index).index(PTN)+1) if PTN in dg.index else 'NA'} of {len(dg)}")
print(f"  Degree vs detection prevalence (Spearman): {spearmanr(deg, prev.values[keep])[0]:.2f}; vs mean log expression: {spearmanr(deg, np.log1p(C.values[:, keep]).mean(0))[0]:.2f}")
# null for the full screen
nulls = []
for b in range(3):
    Rb = spearman_matrix(C.values[:, keep], Mdup.values[rng.permutation(n)]); pb = rho_to_p(Rb, n).ravel()
    ob = np.argsort(pb); qs = pb[ob] * m / (np.arange(m) + 1); qs = np.minimum.accumulate(qs[::-1])[::-1]
    qb = np.empty(m); qb[ob] = np.minimum(qs, 1)
    nulls.append(int(((qb.reshape(Rb.shape) < .05) & (np.abs(Rb) >= .3)).sum()))
print(f"  permutation null (3 reps) edges: {nulls}")
pd.DataFrame({"circRNA": dg.index, "degree": dg.values, "prevalence": prev[dg.index].values}).to_csv(f"{OUT}/results/E2_unbiased_screen_degree.csv", index=False)

# ---------- 3. depth adjustment ----------
print("\n=== Technical-depth adjustment (partial Spearman controlling for per-sample #detected circRNAs)")
depth = (C > 0).sum(1).values.astype(float)
def resid(Mx, Z):
    Zc = np.column_stack([np.ones(len(Z)), Z]); B = np.linalg.lstsq(Zc, Mx, rcond=None)[0]; return Mx - Zc @ B
ixm, iym = design(C, Mdup)
Xr = np.apply_along_axis(rankdata, 0, C.values[:, ixm]); Yr = np.apply_along_axis(rankdata, 0, Mdup.values[:, iym])
xn = C.columns[ixm].values; yn = Mdup.columns[iym].values
base_e, _ = edges_from(spearman_matrix(C.values[:, ixm], Mdup.values[:, iym]), n, xn, yn)
bset = set(zip(base_e.circ, base_e.feat))
from sklearn.decomposition import PCA
mpc = PCA(2).fit_transform((np.log2(Mdup.clip(lower=0) + 1) - np.log2(Mdup.clip(lower=0) + 1).mean()).values / 1.0)
adj_rows = []
for name, Z in [("depth only", depth[:, None]), ("depth + mRNA PC1-2 (stringent)", np.column_stack([depth, mpc]))]:
    A = resid(Xr, Z); B_ = resid(Yr, Z)
    A = (A - A.mean(0)) / (A.std(0) + 1e-12); B_ = (B_ - B_.mean(0)) / (B_.std(0) + 1e-12)
    Rp = A.T @ B_ / n; df_eff = n - 2 - Z.shape[1]
    pp = rho_to_p(Rp, df_eff + 2); qp = bh(pp.ravel()).reshape(pp.shape)
    ii, jj = np.where((qp < .05) & (np.abs(Rp) >= .3))
    e = pd.DataFrame({"circ": xn[ii], "feat": yn[jj], "rho": Rp[ii, jj]}); s = set(zip(e.circ, e.feat))
    ptn = e[e.circ == PTN]
    ptn_ptn = Rp[list(xn).index(PTN), list(yn).index("PTN")] if "PTN" in yn else np.nan
    print(f"  {name:32s}: edges {len(e)} (original {len(base_e)}; retained {len(s & bset)} = {100*len(s & bset)/len(bset):.0f}%); circ_PTN edges {len(ptn)}; circ_PTN-PTN partial rho={ptn_ptn:.3f}")
    adj_rows.append(dict(adjustment=name, n_edges=len(e), retained_of_original=len(s & bset), pct_retained=100 * len(s & bset) / len(bset), circ_PTN_edges=len(ptn), circPTN_PTN_partial_rho=ptn_ptn))
    top = pd.Series([c for c in e.circ]).value_counts().head(5)
    print("      top circRNA by degree after adjustment:", [(i.replace('circ_', ''), int(v)) for i, v in top.items()])
pd.DataFrame(adj_rows).to_csv(f"{OUT}/results/E3_depth_adjustment.csv", index=False)

# ---------- 4. host gene ----------
print("\n=== Host-gene co-variation")
ann = D["circ_ann"].loc[C.columns]
host = ann.GENE_SYMBOL.values
rhos = []; rnd = []
Mg = Mdup
for j, c in enumerate(C.columns):
    if prev[c] < .2 or host[j] not in Mg.columns: continue
    rhos.append(spearmanr(C[c], Mg[host[j]])[0])
    rnd.append(spearmanr(C[c], Mg.iloc[:, rng.integers(Mg.shape[1])])[0])
rhos = np.array(rhos); rnd = np.array(rnd)
print(f"  circRNAs (prevalence>=20%) with host gene measured: {len(rhos)}; median rho(circ,host mRNA)={np.median(rhos):.2f}; fraction rho>0 = {np.mean(rhos>0):.2f}; fraction rho>=0.3 = {np.mean(rhos>=.3):.2f}")
print(f"  random circ-gene pairs: median rho={np.median(rnd):.2f}; fraction rho>=0.3 = {np.mean(rnd>=.3):.3f}; Mann-Whitney p={stats.mannwhitneyu(rhos,rnd).pvalue:.1e}")
# circ_PTN: protein association independent of mRNA?
def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    return spearmanr(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))
pp_ = partial_spearman(C[PTN], D["prot"]["PTN|PTN"], Mdup["PTN"])
print(f"  circ_PTN ~ PTN protein | PTN mRNA: partial rho={pp_[0]:.3f}, p={pp_[1]:.2f}  (raw rho circ-protein 0.511; mRNA-protein rho={spearmanr(Mdup['PTN'],D['prot']['PTN|PTN'])[0]:.2f})")
hostfrac = pd.DataFrame({"circ": [c for c in C.columns if prev[c] >= .2 and host[list(C.columns).index(c)] in Mg.columns], "rho_host": rhos})
hostfrac.to_csv(f"{OUT}/results/E4_circ_host_rho.csv", index=False)

# ---------- 4b. 'protein correlation stronger than mRNA' claim ----------
print("\n=== 'circRNA-protein correlation stronger than circRNA-mRNA correlation' (post-transcriptional claim)")
def boot_diff(c, g, B=2000):
    x = C[c].values; ym = Mdup[g].values; yp = D["prot"][g + "|" + g].values
    d = []
    for b in range(B):
        i = rng.integers(0, n, n)
        d.append(abs(spearmanr(x[i], yp[i])[0]) - abs(spearmanr(x[i], ym[i])[0]))
    obs = abs(spearmanr(x, yp)[0]) - abs(spearmanr(x, ym)[0])
    return obs, np.percentile(d, [2.5, 97.5]), spearmanr(x, ym)[0], spearmanr(x, yp)[0]
cp = pd.read_csv(f"{OUT}/results/B1_reproduced_circRNA_protein_edges.csv").rename(columns={"feature1": "circ", "feature2": "feat", "correlation": "rho"})
tests = []
for c, g in [(PTN, "ELOVL2"), ("circ_chr11_33286413_33287511_HIPK3", "PSPH")]:
    if g in Mdup.columns and (g + "|" + g) in D["prot"].columns:
        obs, ci, rm, rp = boot_diff(c, g)
        print(f"  {c.split('_')[-1]}-{g}: rho_mRNA={rm:.3f}, rho_protein={rp:.3f}; |rho_prot|-|rho_mRNA|={obs:.3f}, 95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}] -> {'CI includes 0 (no evidence of difference)' if ci[0] <= 0 <= ci[1] else 'difference supported'}")
# genome-wide: among significant circ-protein edges with matching mRNA, fraction where protein stronger and test via bootstrap on subset
sub = cp.copy(); sub["gene"] = sub.feat.str.split("|").str[0]
sub = sub[sub.gene.isin(Mdup.columns)]
sub["rho_mrna"] = [spearmanr(C[c], Mdup[g])[0] for c, g in zip(sub.circ, sub.gene)]
sub["diff"] = sub.rho.abs() - sub.rho_mrna.abs()
print(f"  {len(sub)} significant circRNA-protein edges have a measured mRNA; protein stronger in {100*(sub['diff']>0).mean():.0f}%; median diff={sub['diff'].median():.3f}; selection note: edges were selected for significance in the PROTEIN layer, which biases this comparison toward 'protein stronger'")
# unbiased version: all circ-protein pairs of the design
ixp, iyp = design(C, D["prot"])
Rp_all = spearman_matrix(C.values[:, ixp], D["prot"].values[:, iyp]); pn = D["prot"].columns[iyp].str.split("|").str[0]
gm = [g in Mdup.columns for g in pn]
Rm_all = spearman_matrix(C.values[:, ixp], Mdup[[g for g, o in zip(pn, gm) if o]].values)
Rp_m = Rp_all[:, np.array(gm)]
print(f"  Over ALL 200 x {Rp_m.shape[1]} circRNA-gene pairs (no significance selection): |rho_prot| > |rho_mRNA| in {100*np.mean(np.abs(Rp_m)>np.abs(Rm_all)):.1f}% (expected ~50% under no difference)")
sub.to_csv(f"{OUT}/results/E5_protein_vs_mRNA_strength.csv", index=False)

# ---------- 5. positive control + quoted numbers ----------
print("\n=== Positive control: mRNA vs protein of the same gene (n=98)")
genes = [g.split("|")[0] for g in D["prot"].columns]
common = [(g, c) for g, c in zip(genes, D["prot"].columns) if g in Mdup.columns]
rr = np.array([spearmanr(Mdup[g], D["prot"][c])[0] for g, c in common])
print(f"  genes with both mRNA and protein: {len(rr)}; median Spearman = {np.nanmedian(rr):.2f}; fraction positive = {np.mean(rr>0):.2f}; fraction rho>=0.3 = {np.mean(rr>=.3):.2f}")
print("\n=== Numbers quoted in the manuscript vs recomputation (Spearman, n=98)")
chk = [("circ_PTN", PTN, "NES", "mRNA"), ("circ_PTN", PTN, "PTPRZ1", "mRNA"), ("circ_PTN", PTN, "FABP7", "mRNA"), ("circ_PTN", PTN, "PTN", "mRNA"),
       ("circ_PTN", PTN, "PTN|PTN", "protein"), ("circ_PTN", PTN, "ELOVL2|ELOVL2", "protein"), ("circ_HIPK3", "circ_chr11_33286413_33287511_HIPK3", "PSPH|PSPH", "protein")]
for nm, c, g, layer in chk:
    y = Mdup[g] if layer == "mRNA" else D["prot"][g]
    r_, p_ = spearmanr(C[c], y)
    print(f"  {nm} vs {g.split('|')[0]} {layer}: rho={r_:.3f}  p={p_:.2e}")
