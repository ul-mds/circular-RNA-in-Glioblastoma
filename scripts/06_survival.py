"""
F. Survival / clinical associations (Kaplan-Meier, log-rank, Cox) - exploratory
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from scipy import stats
from common import *

D = load(); S = D["samples"]; n = len(S)
C, Mx, P = D["circ"], D["mrna"].loc[:, ~D["mrna"].columns.duplicated()], D["prot"]
clin = D["clinical"].loc[S].copy()
clin["time"] = np.where(clin.PATH_DIAG_TO_DEATH_DAYS.notna(), clin.PATH_DIAG_TO_DEATH_DAYS, clin.PATH_DIAG_TO_LAST_CONTACT_DAYS)
clin["event"] = clin.PATH_DIAG_TO_DEATH_DAYS.notna().astype(int)
clin["male"] = (clin.SEX == "Male").astype(int)
mut = pd.read_csv(f"{ROOT}/mutations.txt", sep="\t", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"], low_memory=False)
nonsil = mut[~mut.Variant_Classification.isin(["Silent", "Intron", "3'UTR", "5'UTR", "5'Flank", "RNA", "3'Flank", "IGR"])]
for g in ["IDH1", "TP53", "PTEN", "EGFR", "NF1"]:
    clin[g] = clin.index.isin(nonsil[nonsil.Hugo_Symbol == g].Tumor_Sample_Barcode).astype(int)
clin["depth"] = (C > 0).sum(1).values
sig = pd.read_csv(f"{DERIVED}/Supp.20_Biomarker_signature_scores_biomarker_signature_scores.csv").set_index("sample_id")["signature_score"]
clin["signature_score"] = sig.reindex(S).values
nmf = pd.read_csv(f"{DERIVED}/Supp.16_Cluster_assignments_cluster_assignments.csv").set_index("sample_id")["nmf_subtype"]
clin["nmf_subtype1"] = nmf.reindex(S).values
ok = clin.time.notna() & (clin.time > 0)
print(f"Samples in the 98-sample set: {n}; with usable follow-up: {int(ok.sum())}; deaths: {int(clin.loc[ok,'event'].sum())}; censored: {int((clin.loc[ok,'event']==0).sum())}")
print(f"Median follow-up (reverse KM) / median OS (KM): ", end="")
kmf = KaplanMeierFitter().fit(clin.loc[ok, "time"], clin.loc[ok, "event"]); print(f"median OS = {kmf.median_survival_time_:.0f} days")
print("Deaths within 30 days of diagnosis (likely perioperative / not tumour-biology related):", int(((clin.time < 30) & (clin.event == 1) & ok).sum()))
print("Mutated (non-silent) in cohort: ", {g: int(clin[g].sum()) for g in ["IDH1", "TP53", "PTEN", "EGFR", "NF1"]}, "| age median %.0f" % clin.AGE.median())

feat = pd.DataFrame(index=S)
PTN = "circ_chr7_137251230_137254974_PTN"; VCAN = "circ_chr5_83537007_83542268_VCAN"; HIPK3 = "circ_chr11_33286413_33287511_HIPK3"
for nm, c in [("circ_PTN", PTN), ("circ_VCAN", VCAN), ("circ_HIPK3", HIPK3)]:
    feat[nm] = np.log2(C[c] + 1)
for g in ["PTN", "VCAN", "HIPK3"]:
    feat[g + "_mRNA"] = np.log2(Mx[g].clip(lower=0) + 1)
feat["PTN_protein"] = P["PTN|PTN"]; feat["VCAN_protein"] = P["VCAN|VCAN"] if "VCAN|VCAN" in P.columns else np.nan
feat["signature_score"] = clin.signature_score; feat["circ_depth(#detected)"] = clin.depth; feat["NMF_subtype1(n=3)"] = clin.nmf_subtype1
feat["age(per 10y)"] = clin.AGE / 10; feat["male"] = clin.male; feat["IDH1_mut"] = clin.IDH1

def cox(cols, data, label=""):
    d = data[cols + ["time", "event"]].dropna(); d = d[d.time > 0]
    cph = CoxPHFitter(penalizer=0.0).fit(d, "time", "event")
    return cph, d

def zs(x): return (x - x.mean()) / x.std()
base = pd.concat([feat, clin[["time", "event"]]], axis=1)[ok.values]
print("\n=== Univariable Cox (continuous, per 1 SD; age per 10 y) ===")
rows = []
for c in ["circ_PTN", "circ_VCAN", "circ_HIPK3", "PTN_mRNA", "VCAN_mRNA", "HIPK3_mRNA", "PTN_protein", "signature_score", "circ_depth(#detected)", "NMF_subtype1(n=3)", "age(per 10y)", "male", "IDH1_mut"]:
    d = base[[c, "time", "event"]].dropna().copy()
    if c not in ("NMF_subtype1(n=3)", "male", "IDH1_mut", "age(per 10y)"): d[c] = zs(d[c])
    try:
        cph = CoxPHFitter().fit(d, "time", "event"); r = cph.summary.loc[c]
        rows.append(dict(variable=c, n=len(d), HR=r["exp(coef)"], lo=r["exp(coef) lower 95%"], hi=r["exp(coef) upper 95%"], p=r["p"], cindex=cph.concordance_index_))
    except Exception as e:
        rows.append(dict(variable=c, n=len(d), HR=np.nan, p=np.nan, note=str(e)[:60]))
U = pd.DataFrame(rows); print(U.round(3).to_string(index=False)); U.to_csv(f"{OUT}/results/F1_univariable_cox.csv", index=False)

print("\n=== Multivariable Cox (circ_PTN adjusted) ===")
mv = []
for label, cols in [("circ_PTN + age + sex", ["circ_PTN", "age(per 10y)", "male"]),
                    ("circ_PTN + age + sex + IDH1", ["circ_PTN", "age(per 10y)", "male", "IDH1_mut"]),
                    ("circ_PTN + PTN_mRNA + age + sex", ["circ_PTN", "PTN_mRNA", "age(per 10y)", "male"]),
                    ("circ_PTN + depth + age + sex", ["circ_PTN", "circ_depth(#detected)", "age(per 10y)", "male"]),
                    ("signature + age + sex", ["signature_score", "age(per 10y)", "male"])]:
    d = base[cols + ["time", "event"]].dropna().copy()
    for c in cols:
        if c in ("circ_PTN", "PTN_mRNA", "signature_score", "circ_depth(#detected)"): d[c] = zs(d[c])
    cph = CoxPHFitter().fit(d, "time", "event"); s = cph.summary
    print(f"  {label}: n={len(d)}, C-index={cph.concordance_index_:.3f}")
    for v in cols:
        print(f"      {v:26s} HR={s.loc[v,'exp(coef)']:.2f} [{s.loc[v,'exp(coef) lower 95%']:.2f}-{s.loc[v,'exp(coef) upper 95%']:.2f}] p={s.loc[v,'p']:.3f}")
        mv.append(dict(model=label, variable=v, HR=s.loc[v, "exp(coef)"], lo=s.loc[v, "exp(coef) lower 95%"], hi=s.loc[v, "exp(coef) upper 95%"], p=s.loc[v, "p"]))
pd.DataFrame(mv).to_csv(f"{OUT}/results/F2_multivariable_cox.csv", index=False)

print("\n=== Sensitivity: excluding deaths <30 days ===")
b2 = base[~((base.time < 30) & (base.event == 1))]
for c in ["circ_PTN", "circ_VCAN", "circ_HIPK3", "signature_score"]:
    d = b2[[c, "age(per 10y)", "male", "time", "event"]].dropna().copy(); d[c] = zs(d[c])
    s = CoxPHFitter().fit(d, "time", "event").summary.loc[c]
    print(f"  {c:16s} n={len(d)} adj. HR={s['exp(coef)']:.2f} [{s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}] p={s['p']:.3f}")

print("\n=== Kaplan-Meier, median split (log-rank) ===")
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5)); km_rows = []
for a, c in zip(ax, ["circ_PTN", "circ_VCAN", "circ_HIPK3", "signature_score"]):
    d = base[[c, "time", "event"]].dropna(); hi = d[c] >= d[c].median()
    lr = logrank_test(d.time[hi], d.time[~hi], d.event[hi], d.event[~hi])
    for m, lab in [(hi, "high"), (~hi, "low")]:
        KaplanMeierFitter().fit(d.time[m], d.event[m], label=f"{lab} (n={m.sum()}, events={int(d.event[m].sum())})").plot_survival_function(ax=a, ci_show=False)
    a.set_title(f"{c}: log-rank p={lr.p_value:.3f}"); a.set_xlabel("days from diagnosis")
    km_rows.append(dict(variable=c, logrank_p=lr.p_value))
    print(f"  {c:16s} log-rank p={lr.p_value:.3f}")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/F2_KM_median_split.png", dpi=120); plt.close()
pd.DataFrame(km_rows).to_csv(f"{OUT}/results/F3_logrank.csv", index=False)

print("\n=== Proportional-hazards check (Schoenfeld), circ_PTN + age + sex ===")
d = base[["circ_PTN", "age(per 10y)", "male", "time", "event"]].dropna().copy(); d["circ_PTN"] = zs(d["circ_PTN"])
cph = CoxPHFitter().fit(d, "time", "event"); ph = proportional_hazard_test(cph, d, time_transform="rank")
print(ph.summary[["test_statistic", "p"]].round(3).to_string())

print("\n=== Exploratory genome-wide Cox: 782 circRNAs detected in >=20% samples (age+sex adjusted) ===")
prev = (C > 0).mean(); keep = prev[prev >= .2].index
gw = []
Zc = np.log2(C[keep] + 1)
for c in keep:
    d = pd.DataFrame({"x": zs(Zc[c]).values, "age": clin.AGE.values / 10, "male": clin.male.values, "time": clin.time.values, "event": clin.event.values}, index=S)[ok.values].dropna()
    try:
        s = CoxPHFitter().fit(d, "time", "event").summary.loc["x"]
        gw.append(dict(circRNA=c, HR=s["exp(coef)"], lo=s["exp(coef) lower 95%"], hi=s["exp(coef) upper 95%"], p=s["p"]))
    except Exception:
        pass
G = pd.DataFrame(gw); G["fdr"] = bh(G.p.values); G = G.sort_values("p")
print(f"  tested {len(G)}; nominal p<0.05: {int((G.p<.05).sum())} (expected by chance ~{.05*len(G):.0f}); FDR<0.10: {int((G.fdr<.10).sum())}; FDR<0.05: {int((G.fdr<.05).sum())}")
print(G.head(6).assign(circRNA=lambda d: d.circRNA.str.replace("circ_", "")).round(3).to_string(index=False))
# depth-adjusted
gw2 = []
for c in keep:
    d = pd.DataFrame({"x": zs(Zc[c]).values, "depth": zs(pd.Series(clin.depth.values)).values, "age": clin.AGE.values / 10, "male": clin.male.values, "time": clin.time.values, "event": clin.event.values})[ok.values].dropna()
    try:
        s = CoxPHFitter().fit(d, "time", "event").summary.loc["x"]; gw2.append(dict(circRNA=c, p=s["p"]))
    except Exception: pass
G2 = pd.DataFrame(gw2); G2["fdr"] = bh(G2.p.values)
print(f"  depth-adjusted: nominal p<0.05: {int((G2.p<.05).sum())}; FDR<0.10: {int((G2.fdr<.10).sum())}")
G.to_csv(f"{OUT}/results/F4_genomewide_cox_circRNA.csv", index=False)
for nm, c in [("circ_PTN", PTN), ("circ_VCAN", VCAN), ("circ_HIPK3", HIPK3)]:
    r = G[G.circRNA == c]
    print(f"  {nm}: in genome-wide list HR={r.HR.values[0]:.2f}, p={r.p.values[0]:.3f}, rank {int(list(G.circRNA).index(c)+1)}/{len(G)}")
