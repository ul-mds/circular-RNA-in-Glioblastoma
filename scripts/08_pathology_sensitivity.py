"""
H. Sensitivity analysis excluding the 7 samples that failed expert pathology review (low tumour nuclei).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from common import *

D = load(); S = D["samples"]; C = D["circ"]; M = D["mrna"].loc[:, ~D["mrna"].columns.duplicated()]; P = D["prot"]
clin = D["clinical"].loc[S]
keep = (clin.EXPERT_PATH_REVIEW == "PASS").values
print("samples passing expert pathology review:", keep.sum(), "of", len(S))
PTN = "circ_chr7_137251230_137254974_PTN"
rows = []
for lab, X in [("PTN mRNA", M["PTN"]), ("PTN protein", P["PTN|PTN"])]:
    r_all = spearmanr(C[PTN], X)[0]; r_k = spearmanr(C.loc[keep, PTN], X[keep])[0]
    rows.append({"analysis": f"circ_PTN vs {lab} (Spearman rho)", "all_samples": r_all, "pass_only": r_k})
    print(f"circ_PTN vs {lab}: all rho={r_all:.3f}, pass-only rho={r_k:.3f}")
# host-gene co-variation across all circRNAs with prevalence>=20%
prev = (C > 0).mean(); cols = [c for c in C.columns if prev[c] >= .2]
def host_rho(mask):
    out = []
    for c in cols:
        g = c.split("_")[-1]
        if g in M.columns:
            out.append(spearmanr(C.loc[mask, c], M.loc[mask, g])[0])
    return np.array(out)
ra = host_rho(np.ones(len(S), bool)); rk = host_rho(keep)
print(f"host-gene rho median: all={np.median(ra):.2f} (n={len(ra)}), pass-only={np.median(rk):.2f}")
rows.append({"analysis": "median host-gene rho, circRNAs with >=20% prevalence", "all_samples": np.median(ra), "pass_only": np.median(rk)})
# survival
t = np.where(clin.PATH_DIAG_TO_DEATH_DAYS.notna(), clin.PATH_DIAG_TO_DEATH_DAYS, clin.PATH_DIAG_TO_LAST_CONTACT_DAYS)
e = clin.PATH_DIAG_TO_DEATH_DAYS.notna().astype(int).values
df = pd.DataFrame({"t": t, "e": e, "circ_PTN": np.log2(C[PTN].values + 1), "age": clin.AGE.values / 10, "male": (clin.SEX == "Male").astype(int).values}, index=S)
df = df[(df.t > 0)]
df["circ_PTN"] = (df.circ_PTN - df.circ_PTN.mean()) / df.circ_PTN.std()
for lab, d in [("all", df), ("pass-only", df[(clin.EXPERT_PATH_REVIEW == "PASS").reindex(df.index)])]:
    d = d.copy(); d["circ_PTN"] = (d.circ_PTN - d.circ_PTN.mean()) / d.circ_PTN.std()
    f = CoxPHFitter().fit(d, "t", "e"); s = f.summary.loc["circ_PTN"]
    print(f"Cox circ_PTN + age + sex ({lab}, n={len(d)}, events={int(d.e.sum())}): HR={np.exp(s['coef']):.2f} [{np.exp(s['coef lower 95%']):.2f}-{np.exp(s['coef upper 95%']):.2f}] p={s['p']:.2f}")
    rows.append({"analysis": f"circ_PTN HR per SD, adjusted for age and sex ({lab}; n={len(d)})", "all_samples": np.exp(s['coef']) if lab=='all' else np.nan, "pass_only": np.exp(s['coef']) if lab!='all' else np.nan})
pd.DataFrame(rows).to_csv(f"{OUT}/results/H1_pathology_sensitivity.csv", index=False)
