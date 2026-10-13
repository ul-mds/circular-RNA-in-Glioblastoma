"""
G. Independent circRNA-level check in GSE196695 (primary/recurrent GBM vs healthy brain; rRNA-depleted RNA-seq, CIRI2, GRCh37)
   - CPTAC circRNA coordinates are GRCh38 -> lifted to GRCh37 (+-2 nt tolerance for BSJ coordinate conventions)
   - replication rate of CPTAC circRNAs (by detection prevalence), abundance concordance
   - specific circRNAs of the manuscript (hubs / ceRNA pairs / GRB10): detected in GBM vs healthy brain?
"""
import warnings; warnings.filterwarnings("ignore")
import re, glob
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from pyliftover import LiftOver
from common import *

D = load(); C = D["circ"]
EXT = f"{OUT}/cache/ext"
rows = []
for f in sorted(glob.glob(f"{EXT}/raw/*_circ.txt.gz")):
    name = re.sub(r"^GSM\d+_|_circ\.txt\.gz$", "", f.split("/")[-1])
    d = pd.read_csv(f, sep="\t", usecols=["chr", "circRNA_start", "circRNA_end", "#junction_reads", "#non_junction_reads", "junction_reads_ratio", "circRNA_type"])
    d["sample"] = name; rows.append(d)
A = pd.concat(rows); A.columns = ["chr", "start", "end", "jr", "njr", "ratio", "type", "sample"]
grp = lambda s: "healthy_brain" if s.startswith("hb") else "primary_GBM" if re.match(r"P\d+$", s) else "recurrent_GBM" if re.match(r"R\d+$", s) else "blood"
A["group"] = A["sample"].map(grp)
samples = A.groupby("sample")["group"].first()
print("External samples:", samples.value_counts().to_dict())
qc = A.groupby(["group", "sample"]).agg(n_circ=("jr", "size"), total_bsj_reads=("jr", "sum")).reset_index()
print("\nExternal per-group QC (number of distinct BSJ per sample; median [min-max]):")
for g, x in qc.groupby("group"):
    print(f"  {g:14s} n={len(x):2d}  circRNAs/sample median {x.n_circ.median():.0f} [{x.n_circ.min()}-{x.n_circ.max()}];  BSJ reads median {x.total_bsj_reads.median():.0f}")
tum = [s for s, g in samples.items() if g in ("primary_GBM", "recurrent_GBM")]
hb = [s for s, g in samples.items() if g == "healthy_brain"]

key = A.assign(k=A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str))
pivot_jr = key.pivot_table(index="k", columns="sample", values="jr", aggfunc="sum", fill_value=0)
pivot_ratio = key.pivot_table(index="k", columns="sample", values="ratio", aggfunc="max", fill_value=0)
ext_index = {}
for k in pivot_jr.index:
    c, se = k.split(":"); s, e = map(int, se.split("-")); ext_index.setdefault(c, {})[(s, e)] = k

lo = LiftOver(f"{OUT}/cache/ref/hg38ToHg19.over.chain.gz")
def lift(c, p):
    r = lo.convert_coordinate(c, p)
    return r[0][1] if r else None
def match(cid, tol=2):
    m = re.match(r"circ_(chr\w+?)_(\d+)_(\d+)_(.+)$", cid); c, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    s19, e19 = lift(c, s), lift(c, e)
    if s19 is None or e19 is None: return None
    best = None
    for ds in range(-tol, tol + 1):
        for de in range(-tol, tol + 1):
            k = ext_index.get(c, {}).get((s19 + ds, e19 + de))
            if k is not None:
                dist = abs(ds) + abs(de)
                if best is None or dist < best[0]: best = (dist, k)
    return best

M = {cid: match(cid) for cid in C.columns}
matched = {c: v for c, v in M.items() if v is not None}
print(f"\nCPTAC circRNAs with an exact-junction BSJ in GSE196695 (liftover hg38->hg19, +-2 nt): {len(matched)} / {len(M)}")
print("   of which exact (0 nt):", sum(1 for v in matched.values() if v[0] == 0), "; within 1-2 nt:", sum(1 for v in matched.values() if v[0] > 0))

prev = (C > 0).mean()
res = []
for cid in C.columns:
    v = M[cid]
    if v is None:
        res.append(dict(circRNA=cid, prevalence=prev[cid], matched=False)); continue
    k = v[1]
    jt = pivot_jr.loc[k, tum]; jh = pivot_jr.loc[k, hb]
    res.append(dict(circRNA=cid, prevalence=prev[cid], matched=True, nt_offset=v[0], ext_key=k,
                    det_GBM=int((jt > 0).sum()), det_HB=int((jh > 0).sum()), n_GBM=len(tum), n_HB=len(hb),
                    mean_jr_GBM=jt.mean(), mean_jr_HB=jh.mean(), mean_ratio_GBM=pivot_ratio.loc[k, tum].mean(), mean_ratio_HB=pivot_ratio.loc[k, hb].mean()))
R = pd.DataFrame(res); R.to_csv(f"{OUT}/results/G1_external_matching_all_circRNAs.csv", index=False)
bins = [(0, .05), (.05, .2), (.2, .5), (.5, 1.01)]
print("\nReplication of CPTAC circRNAs in external GBM (>=1 of 29 tumours), by CPTAC detection prevalence:")
for lo_, hi_ in bins:
    sub = R[(R.prevalence >= lo_) & (R.prevalence < hi_)]
    mm = sub[sub.matched]
    print(f"  prevalence [{lo_:.2f},{min(hi_,1):.2f}): n={len(sub):4d}  BSJ found in external GBM: {int((mm.det_GBM > 0).sum()):4d} ({100*(mm.det_GBM>0).sum()/len(sub):5.1f}%);  in >=25% of GBM tumours: {int((mm.det_GBM >= 8).sum()):4d} ({100*(mm.det_GBM>=8).sum()/len(sub):5.1f}%)")

mm = R[R.matched & (R.det_GBM > 0)]
cp_mean = C[mm.circRNA].mean().values
print(f"\nAbundance concordance (matched circRNAs detected in external GBM, n={len(mm)}): Spearman rho (CPTAC mean expression vs external mean BSJ reads) = {spearmanr(cp_mean, mm.mean_jr_GBM)[0]:.2f}; vs mean ratio = {spearmanr(cp_mean, mm.mean_ratio_GBM)[0]:.2f}")
cpt_prev = prev[mm.circRNA].values
print(f"   prevalence concordance: rho(CPTAC prevalence, external GBM detection fraction) = {spearmanr(cpt_prev, mm.det_GBM / mm.n_GBM)[0]:.2f}")

# ---- the circRNAs that the manuscript highlights
focus = {"PTN": "circ_chr7_137251230_137254974_PTN", "VCAN": "circ_chr5_83537007_83542268_VCAN", "HIPK3": "circ_chr11_33286413_33287511_HIPK3",
         "RNF220": "circ_chr1_44411981_44412722_RNF220", "DNAJC6": "circ_chr1_65364635_65366196_DNAJC6", "FAT3": "circ_chr11_92352096_92355404_FAT3",
         "ATRNL1": "circ_chr10_115120185_115171292_ATRNL1", "EDIL3": "circ_chr5_84060300_84066606_EDIL3",
         "GRB10(top RF)": "circ_chr7_50669722_50703908_GRB10", "ZNF91 (top degree)": "circ_chr19_23358430_23362725_ZNF91", "RERE (2nd degree)": "circ_chr1_8541214_8557523_RERE"}
print("\nManuscript-highlighted circRNAs in GSE196695 (detected samples / n):  GBM(29)  healthy brain(18)")
out = []
for nm, cid in focus.items():
    r = R[R.circRNA == cid].iloc[0]
    if not r.matched:
        # report nearest BSJ in same gene region for context
        print(f"  {nm:20s} CPTAC prevalence {r.prevalence:.2f}: NO matching BSJ in external data"); out.append(dict(circRNA=nm, matched=False)); continue
    k = r.ext_key
    a, b = int(r.det_GBM), int(r.det_HB)
    p_f = fisher_exact([[a, r.n_GBM - a], [b, r.n_HB - b]])[1]
    jt = pivot_jr.loc[k, tum].values; jh = pivot_jr.loc[k, hb].values
    pm = mannwhitneyu(jt, jh).pvalue if (jt.sum() + jh.sum()) > 0 else np.nan
    rt = pivot_ratio.loc[k, tum].values; rh = pivot_ratio.loc[k, hb].values   # junction ratio is normalised for library depth
    pr = mannwhitneyu(rt, rh).pvalue if (rt.sum() + rh.sum()) > 0 else np.nan
    print(f"      junction ratio: GBM mean {rt.mean():.3f} vs brain mean {rh.mean():.3f} (MWU p={pr:.3g})")
    print(f"  {nm:20s} CPTAC prev {r.prevalence:.2f} | ext BSJ {k} (offset {int(r.nt_offset)} nt): GBM {a:2d}/{int(r.n_GBM)}, brain {b:2d}/{int(r.n_HB)} (Fisher p={p_f:.3g}); mean BSJ reads GBM {r.mean_jr_GBM:.2f} vs brain {r.mean_jr_HB:.2f} (MWU p={pm:.3g})")
    out.append(dict(circRNA=nm, matched=True, ext_key=k, det_GBM=a, det_HB=b, fisher_p=p_f, mean_jr_GBM=r.mean_jr_GBM, mean_jr_HB=r.mean_jr_HB, mwu_p=pm, mean_ratio_GBM=rt.mean(), mean_ratio_HB=rh.mean(), mwu_p_ratio=pr))
pd.DataFrame(out).to_csv(f"{OUT}/results/G2_external_focus_circRNAs.csv", index=False)

# PTN locus: any BSJ at all in the PTN locus in external data (hg19 chr7:136,912,091-137,028,479)?
loc = key[(key.chr == "chr7") & (key.start >= 136912091) & (key.end <= 137028479)]
if len(loc):
    g = loc.groupby("k").agg(n=("sample", "nunique"), reads=("jr", "sum")).sort_values("n", ascending=False)
    print("\nBSJs inside the PTN locus (hg19) in external data:\n", g.head(5).to_string())
else:
    print("\nNo BSJ at all inside the PTN locus (hg19 chr7:136,912,091-137,028,479) in GSE196695")

# GBM vs healthy brain among well-detected matched circRNAs
w = R[R.matched & ((R.det_GBM >= 8) | (R.det_HB >= 5))].copy()
w["p"] = [mannwhitneyu(pivot_ratio.loc[k, tum], pivot_ratio.loc[k, hb]).pvalue for k in w.ext_key]
w["fdr"] = bh(w.p.values)
w["lfc"] = np.log2((w.mean_ratio_GBM + 1e-3) / (w.mean_ratio_HB + 1e-3))
print(f"\nGBM vs healthy brain (junction ratio) among {len(w)} well-detected, CPTAC-matched circRNAs: FDR<0.05 higher in GBM = {int(((w.fdr<.05)&(w.lfc>0)).sum())}, higher in brain = {int(((w.fdr<.05)&(w.lfc<0)).sum())}")
w.to_csv(f"{OUT}/results/G3_external_GBM_vs_brain_matched.csv", index=False)
