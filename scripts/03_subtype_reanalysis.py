"""
C. circRNA 'subtype' re-evaluation
 1) reproduce original clustering / silhouette / label sizes
 2) method concordance, outlier / technical-driver analysis
 3) class-specific classifier metrics vs majority baseline (+ permutation test)
 4) robust re-clustering (log, detection-filtered) with consensus-clustering stability
 5) agreement with mRNA- and protein-based clustering
 6) diagnostic figure (PCA / t-SNE at several perplexities)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, adjusted_rand_score, confusion_matrix, precision_recall_fscore_support,
                             balanced_accuracy_score, matthews_corrcoef, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from scipy import stats
from common import *

rng = np.random.default_rng(11)
D = load(); S = D["samples"]; n = len(S)
C = D["circ"]
orig = pd.read_csv(f"{DERIVED}/Supp.16_Cluster_assignments_cluster_assignments.csv").set_index("sample_id").loc[S]

# ---------- 1. reproduce ----------
sc = RobustScaler().fit(C.values)
print("RobustScaler: features with IQR==0 (sklearn falls back to scale=1, i.e. NOT standardised):", int((sc.scale_ == 1).sum()) if False else int(np.sum(np.subtract(*np.percentile(C.values, [75, 25], axis=0)) == 0)), "of", C.shape[1])
Xs = pd.DataFrame(sc.transform(C.values), index=S, columns=C.columns)
Xs = Xs.loc[:, Xs.std() > 0.01]
print("features retained after std>0.01:", Xs.shape[1])
sil = {}
for k in range(2, 6):
    lab = KMeans(k, random_state=42, n_init=10).fit_predict(Xs.values)
    sil[k] = (silhouette_score(Xs.values, lab), np.bincount(lab).tolist())
print("k-means silhouette / cluster sizes (original preprocessing):", {k: (round(v[0], 3), v[1]) for k, v in sil.items()})
km = KMeans(2, random_state=42, n_init=10).fit_predict(Xs.values)
hc = fcluster(linkage(Xs.values, "ward"), 2, "maxclust") - 1
Xn = Xs - Xs.min().min() + 0.1
W = NMF(2, random_state=42, max_iter=1000).fit_transform(Xn.values); nmf = W.argmax(1)
print("re-run labels vs Supp.16 :  kmeans ARI=%.2f hier ARI=%.2f nmf ARI=%.2f" % (
    adjusted_rand_score(orig.kmeans_subtype, km), adjusted_rand_score(orig.hierarchical_subtype, hc), adjusted_rand_score(orig.nmf_subtype, nmf)))
print("sizes  kmeans:", np.bincount(orig.kmeans_subtype).tolist(), " hier:", np.bincount(orig.hierarchical_subtype).tolist(), " nmf:", np.bincount(orig.nmf_subtype).tolist())

# ---------- 2. concordance & outliers ----------
lab = orig.copy()
pairs = [("kmeans_subtype", "hierarchical_subtype"), ("kmeans_subtype", "nmf_subtype"), ("hierarchical_subtype", "nmf_subtype")]
print("\nPairwise ARI between the three methods:", {f"{a[:4]}-{b[:4]}": round(adjusted_rand_score(lab[a], lab[b]), 2) for a, b in pairs})
for m in ["kmeans_subtype", "hierarchical_subtype", "nmf_subtype"]:
    print(f"  {m}: minority-cluster members =", list(lab.index[lab[m] == lab[m].value_counts().idxmin()]))
minor = list(lab.index[lab.nmf_subtype == 1])
qc = pd.DataFrame({"n_detected": (C > 0).sum(1), "total_signal": C.sum(1), "max_single_circ": C.max(1)})
qc["pct_rank_detected"] = qc.n_detected.rank(pct=True) * 100
qc["pct_rank_total"] = qc.total_signal.rank(pct=True) * 100
pca = PCA(10).fit(Xs.values); Z = pca.transform(Xs.values)
qc["PC1"] = Z[:, 0]; qc["PC2"] = Z[:, 1]
d = np.sqrt(((Z[:, :5] - np.median(Z[:, :5], 0)) ** 2).sum(1)); qc["pca_dist_rank_pct"] = pd.Series(d, index=S).rank(pct=True) * 100
print("\nPCA variance explained (original preprocessing) PC1-5:", (pca.explained_variance_ratio_[:5] * 100).round(1).tolist())
print("Spearman PC1 vs #detected circRNAs: rho=%.2f; PC1 vs total signal: rho=%.2f" % (
    stats.spearmanr(Z[:, 0], qc.n_detected)[0], stats.spearmanr(Z[:, 0], qc.total_signal)[0]))
print("\nSubtype-1 (NMF) members, QC percentile ranks among 98 samples:")
print(qc.loc[minor, ["n_detected", "total_signal", "max_single_circ", "pct_rank_detected", "pct_rank_total", "pca_dist_rank_pct"]].round(1).to_string())
clin = D["clinical"].loc[S]
print(clin.loc[minor, ["AGE", "SEX", "VITAL_STATUS", "PATH_DIAG_TO_DEATH_DAYS", "PATH_DIAG_TO_LAST_CONTACT_DAYS"]].to_string())
# features that drive the NMF split: which circRNAs have extreme robust-z in the minority samples
rz = Xs.loc[minor]
print("\nTop circRNAs by robust-z in minority samples (value, cohort max, #samples detected):")
for s in minor:
    top = rz.loc[s].sort_values(ascending=False).head(3)
    print(" ", s, [(c.replace("circ_", ""), round(C.loc[s, c], 1), round(C[c].max(), 1), int((C[c] > 0).sum())) for c in top.index])
qc.to_csv(f"{OUT}/results/C1_sample_QC_and_PCA.csv")
orig_df = pd.DataFrame({"kmeans": orig.kmeans_subtype, "hier": orig.hierarchical_subtype, "nmf": orig.nmf_subtype})
orig_df.join(qc).to_csv(f"{OUT}/results/C1b_labels_with_QC.csv")

# ---------- 3. class-specific metrics ----------
y = orig.nmf_subtype.values
rf = RandomForestClassifier(100, random_state=42, n_jobs=-1)
pred_loo = cross_val_predict(rf, Xs.values, y, cv=LeaveOneOut(), n_jobs=1)
def report(yt, yp, name):
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    print(f"\n[{name}] accuracy={accuracy_score(yt, yp):.4f} balanced_acc={balanced_accuracy_score(yt, yp):.3f} MCC={matthews_corrcoef(yt, yp):.3f}")
    print("   confusion matrix [[TN(0->0), 0->1],[1->0, TP(1->1)]]:", cm.tolist())
    print(f"   class 0: precision={p[0]:.3f} recall(sens)={r[0]:.3f} F1={f[0]:.3f} n={s[0]}")
    print(f"   class 1: precision={p[1]:.3f} recall(sens)={r[1]:.3f} F1={f[1]:.3f} n={s[1]}   specificity(class1)={cm[0,0]/cm[0].sum():.3f}")
    return dict(model=name, accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp), mcc=matthews_corrcoef(yt, yp),
                prec0=p[0], rec0=r[0], f1_0=f[0], prec1=p[1], rec1=r[1], f1_1=f[1], tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1])
rows = [report(y, pred_loo, "RF leave-one-out")]
print("   majority-class baseline accuracy (always predict class 0): %.4f" % (np.mean(y == 0)))
accs = []; preds = []
for rep in range(10):
    pr = cross_val_predict(RandomForestClassifier(100, random_state=rep, n_jobs=-1), Xs.values, y, cv=StratifiedKFold(3, shuffle=True, random_state=rep))
    preds.append(pr)
rows.append(report(np.tile(y, 10), np.concatenate(preds), "RF 3-fold stratified x10 (pooled)"))
# permutation test for balanced accuracy (3-fold stratified, 100 perms)
obs = balanced_accuracy_score(y, preds[0]); null = []
for b in range(100):
    yp_ = rng.permutation(y)
    null.append(balanced_accuracy_score(yp_, cross_val_predict(RandomForestClassifier(100, random_state=b, n_jobs=-1), Xs.values, yp_,
                                                               cv=StratifiedKFold(3, shuffle=True, random_state=b))))
print("   permutation test (3-fold, 100 perms): observed balanced acc=%.3f; null mean=%.3f (max %.3f); p=%.3f" % (obs, np.mean(null), np.max(null), (np.sum(np.array(null) >= obs) + 1) / 101))
pd.DataFrame(rows).to_csv(f"{OUT}/results/C2_class_specific_metrics.csv", index=False)

# ---------- 4. robust re-clustering + consensus ----------
prev = (C > 0).mean()
keep = prev[prev >= .20].index
L = np.log2(C[keep] + 1)
Zs = pd.DataFrame(StandardScaler().fit_transform(L.values), index=S, columns=keep)
print(f"\n[4] Robust preprocessing: log2(x+1), circRNAs detected in >=20% samples ({len(keep)}), z-scored")
res = []
for k in range(2, 7):
    lb = KMeans(k, random_state=0, n_init=20).fit_predict(Zs.values)
    hb = fcluster(linkage(Zs.values, "ward"), k, "maxclust") - 1
    res.append((k, round(silhouette_score(Zs.values, lb), 3), np.bincount(lb).tolist(), round(silhouette_score(Zs.values, hb), 3), np.bincount(hb).tolist()))
for r in res: print("   k=%d  kmeans sil=%.3f sizes=%s | Ward sil=%.3f sizes=%s" % r)
pcr = PCA(10).fit(Zs.values); Zr = pcr.transform(Zs.values)
print("   PCA variance PC1-5 (%):", (pcr.explained_variance_ratio_[:5] * 100).round(1).tolist(), "| PC1 vs #detected rho=%.2f" % stats.spearmanr(Zr[:, 0], qc.n_detected)[0])

def consensus(X, k, H=300, ps=.8, pf=.8):
    M = np.zeros((n, n)); Cn = np.zeros((n, n))
    for h in range(H):
        si = rng.choice(n, int(ps * n), replace=False); fi = rng.choice(X.shape[1], int(pf * X.shape[1]), replace=False)
        l = KMeans(k, n_init=3, random_state=int(rng.integers(1e9))).fit_predict(X[np.ix_(si, fi)])
        same = (l[:, None] == l[None, :]).astype(float)
        M[np.ix_(si, si)] += same; Cn[np.ix_(si, si)] += 1
    return M / np.maximum(Cn, 1)
print("   Consensus clustering (300 subsamples, 80% samples x 80% features):")
cons_rows = []
for k in range(2, 7):
    Cm = consensus(Zs.values, k)
    iu = np.triu_indices(n, 1); v = Cm[iu]
    pac = np.mean((v > .1) & (v < .9))
    cons_rows.append(dict(k=k, PAC=pac, mean_consensus=v.mean()))
    print(f"      k={k}: PAC (ambiguous pairs, lower=more stable) = {pac:.3f}")
pd.DataFrame(cons_rows).to_csv(f"{OUT}/results/C3_consensus_PAC.csv", index=False)

# bootstrap stability of the ORIGINAL pipeline (re-implemented) and of the robust one
def boot_ari(Xfun, k=2, B=100):
    ref = KMeans(k, random_state=0, n_init=10).fit_predict(Xfun(np.arange(n)))
    out = []
    for b in range(B):
        idx = np.sort(rng.choice(n, n, replace=True)); uniq = np.unique(idx)
        l = KMeans(k, random_state=b, n_init=5).fit_predict(Xfun(uniq))
        out.append(adjusted_rand_score(ref[uniq], l))
    return np.array(out)
b_orig = boot_ari(lambda ix: Xs.values[ix]); b_rob = boot_ari(lambda ix: Zs.values[ix])
print(f"   bootstrap ARI of k=2 k-means: original preprocessing mean={b_orig.mean():.2f} (median {np.median(b_orig):.2f}, <0.5 in {100*np.mean(b_orig<.5):.0f}%);  robust preprocessing mean={b_rob.mean():.2f} (median {np.median(b_rob):.2f}, <0.5 in {100*np.mean(b_rob<.5):.0f}%)")
sup9 = pd.read_csv(f"{DERIVED}/Supp.9_Bootstrap_stability_results_bootstrap_stability_results.csv").adjusted_rand_index
print(f"   (published Supp.9 bootstrap ARI: mean={sup9.mean():.2f}, median={sup9.median():.2f}, ARI<0.5 in {100*np.mean(sup9<.5):.0f}% of 100 runs)")

# ---------- 5. mRNA / protein clustering comparison ----------
def top_var(df, k=2000, log=False):
    X = np.log2(df.clip(lower=0) + 1) if log else df
    mad = (X - X.median()).abs().median()
    X = X.loc[:, mad.sort_values(ascending=False).head(k).index]
    return StandardScaler().fit_transform(X.values)
Xm = top_var(D["mrna"].loc[:, ~D["mrna"].columns.duplicated()], log=True)
Xp = top_var(D["prot"], log=False)
print("\n[5] Agreement between circRNA clusters and mRNA / protein clusters (top-2000 MAD features)")
cmp_rows = []
for k in [2, 3, 4]:
    lm = KMeans(k, random_state=0, n_init=20).fit_predict(Xm); lp = KMeans(k, random_state=0, n_init=20).fit_predict(Xp)
    lc = KMeans(k, random_state=0, n_init=20).fit_predict(Zs.values)
    row = dict(k=k, sil_mRNA=silhouette_score(Xm, lm), sil_protein=silhouette_score(Xp, lp), sil_circ=silhouette_score(Zs.values, lc),
               ARI_circ_vs_mRNA=adjusted_rand_score(lc, lm), ARI_circ_vs_protein=adjusted_rand_score(lc, lp), ARI_mRNA_vs_protein=adjusted_rand_score(lm, lp),
               ARI_origNMF_vs_mRNA=adjusted_rand_score(y, lm) if k == 2 else np.nan, ARI_origNMF_vs_protein=adjusted_rand_score(y, lp) if k == 2 else np.nan)
    cmp_rows.append(row)
    print("   k=%d silhouette: circ=%.2f mRNA=%.2f protein=%.2f | ARI circ-mRNA=%.2f circ-protein=%.2f mRNA-protein=%.2f%s" % (
        k, row["sil_circ"], row["sil_mRNA"], row["sil_protein"], row["ARI_circ_vs_mRNA"], row["ARI_circ_vs_protein"], row["ARI_mRNA_vs_protein"],
        (" | orig NMF vs mRNA=%.2f vs protein=%.2f" % (row["ARI_origNMF_vs_mRNA"], row["ARI_origNMF_vs_protein"])) if k == 2 else ""))
pd.DataFrame(cmp_rows).to_csv(f"{OUT}/results/C4_cluster_agreement_mRNA_protein.csv", index=False)
# PCA-level association: circRNA PCs vs mRNA/protein PCs
pm = PCA(5).fit_transform(Xm); pp = PCA(5).fit_transform(Xp)
print("   Spearman |rho| (circ PC1 vs mRNA PC1..3):", [round(abs(stats.spearmanr(Zr[:, 0], pm[:, i])[0]), 2) for i in range(3)],
      "| circ PC1 vs protein PC1..3:", [round(abs(stats.spearmanr(Zr[:, 0], pp[:, i])[0]), 2) for i in range(3)])

# ---------- 6. figure ----------
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
col = np.where(y == 1, "crimson", "steelblue")
ax[0, 0].scatter(Z[:, 0], Z[:, 1], c=col, s=40); ax[0, 0].set_title("PCA, original preprocessing\n(red = NMF 'Subtype 1', n=3)")
ax[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
for i, perp in enumerate([30, 10]):
    T = TSNE(2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(Xs.values)
    ax[0, i + 1].scatter(T[:, 0], T[:, 1], c=col, s=40); ax[0, i + 1].set_title(f"t-SNE, original preprocessing, perplexity {perp}")
sc1 = ax[1, 0].scatter(Zr[:, 0], Zr[:, 1], c=qc.n_detected, cmap="viridis", s=40); plt.colorbar(sc1, ax=ax[1, 0], label="# detected circRNAs")
ax[1, 0].set_title("PCA, robust preprocessing (log2, detection >=20%)\ncolour = # detected circRNAs"); ax[1, 0].set_xlabel("PC1"); ax[1, 0].set_ylabel("PC2")
ax[1, 0].scatter(Zr[np.array([s in minor for s in S]), 0], Zr[np.array([s in minor for s in S]), 1], facecolors="none", edgecolors="red", s=120, linewidths=2)
ax[1, 1].hist(b_orig, bins=20, alpha=.7, label="original preprocessing"); ax[1, 1].hist(b_rob, bins=20, alpha=.7, label="robust preprocessing")
ax[1, 1].set_xlabel("bootstrap ARI vs reference k-means (k=2)"); ax[1, 1].legend(); ax[1, 1].set_title("Cluster stability under resampling")
ax[1, 2].scatter(qc.n_detected, Z[:, 0], c=col, s=40); ax[1, 2].set_xlabel("# detected circRNAs per sample"); ax[1, 2].set_ylabel("PC1 (original preprocessing)")
ax[1, 2].set_title("PC1 tracks detection depth (technical)")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/F1_subtype_diagnostics.png", dpi=130); plt.close()
print("\nfigure saved")
