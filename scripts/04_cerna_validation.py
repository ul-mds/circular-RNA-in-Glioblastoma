"""
D. circRNA-miRNA (ceRNA candidate) re-evaluation:
 1) reproduce the miRNA screen, test sensitivity to median imputation (miRNA layer is heavily imputed)
 2) sequence-level support: spliced circRNA sequence (hg38 refGene exons + UCSC genomic sequence) vs miRNA seed sites
 3) positive control for the statistical approach: do TargetScan-predicted targets of a miRNA show negative correlation with that miRNA in our data?
"""
import gzip, json, re, time, urllib.request
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr
from common import *

rng = np.random.default_rng(7)
D = load(); n = len(D["samples"])
C, MI, MIraw = D["circ"], D["mirna"], D["mirna_raw"]
REF = f"{OUT}/cache/ref"

# ---------------- 1. reproduce screen ----------------
vc = C.var().values; vm = MI.var().values
ic = np.argsort(-vc, kind="stable")[:300]; im = np.argsort(-vm, kind="stable")[:100]
R = spearman_matrix(C.values[:, ic], MI.values[:, im]); p = rho_to_p(R, n); q = bh(p.ravel()).reshape(p.shape)
ii, jj = np.where((q < .05) & (np.abs(R) >= .3))
pairs = pd.DataFrame({"circRNA": C.columns[ic][ii], "miRNA": MI.columns[im][jj], "rho_imputed": R[ii, jj], "fdr": q[ii, jj]})
print(f"[1] circRNA-miRNA screen: 300 x 100 = {R.size} tests; significant pairs = {len(pairs)}")
# sensitivity: pairwise-complete (no imputation)
rows = []
for _, r in pairs.iterrows():
    x = C[r.circRNA]; y = MIraw[r.miRNA]; ok = y.notna()
    rho, pv = spearmanr(x[ok], y[ok])
    rows.append(dict(n_miRNA_observed=int(ok.sum()), pct_miRNA_imputed=100 * (1 - ok.mean()), rho_complete_case=rho, p_complete_case=pv,
                     circ_detect_prev=(x > 0).mean(), mir_n_unique_imputed=y.fillna(y.median()).nunique()))
pairs = pd.concat([pairs, pd.DataFrame(rows)], axis=1)
pairs["sign_expected_for_sponge"] = "negative"
pairs["sign_consistent_with_sponge"] = pairs.rho_imputed < 0
print(pairs.assign(circRNA=lambda d: d.circRNA.str.replace("circ_", "")).round(3).to_string(index=False))
print("Pairs with NEGATIVE rho (sponge-consistent):", int(pairs.sign_consistent_with_sponge.sum()), "of", len(pairs))

# permutation null for the miRNA screen
nul = []
Cv, Mv = C.values[:, ic], MI.values[:, im]
for b in range(200):
    Rb = spearman_matrix(Cv, Mv[rng.permutation(n)]); pb = rho_to_p(Rb, n); qb = bh(pb.ravel()).reshape(pb.shape)
    nul.append(int(((qb < .05) & (np.abs(Rb) >= .3)).sum()))
print(f"   permutation null (200 reps): mean significant pairs = {np.mean(nul):.2f}, max = {max(nul)}, P(null>={len(pairs)}) = {np.mean(np.array(nul) >= len(pairs)):.3f}")
# unadjusted: how many pass p<0.05 & |rho|>=.3 at random
nn = []
for b in range(50):
    Rb = spearman_matrix(Cv, Mv[rng.permutation(n)]); nn.append(int(((rho_to_p(Rb, n) < .05) & (np.abs(Rb) >= .3)).sum()))
print(f"   unadjusted null hits (p<.05 & |rho|>=.3): mean {np.mean(nn):.0f} of {R.size}")

# ---------------- 2. sequence-level support ----------------
def fetch_seq(chrom, start0, end):
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start0};end={end}"
    for t in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as f:
                return json.load(f)["dna"].upper()
        except Exception as e:
            time.sleep(2)
    raise RuntimeError("fetch failed " + url)

rg = pd.read_csv(f"{REF}/refGene_hg38.txt.gz", sep="\t", header=None, compression="gzip",
                 names="bin name chrom strand txStart txEnd cdsStart cdsEnd exonCount exonStarts exonEnds score name2 cdsStartStat cdsEndStat exonFrames".split())

def circ_exons(cid):
    m = re.match(r"circ_(chr\w+?)_(\d+)_(\d+)_(.+)$", cid)
    chrom, s, e, gene = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    sub = rg[(rg.chrom == chrom) & (rg.name2 == gene.split("-")[0] if False else (rg.name2 == gene))]
    best = None
    for _, t in sub.iterrows():
        es = [int(x) for x in t.exonStarts.strip(",").split(",")]; ee = [int(x) for x in t.exonEnds.strip(",").split(",")]
        # exon-boundary agreement score for the back-splice junction
        d_start = min(abs(a + 1 - s) for a in es); d_end = min(abs(b - e) for b in ee)
        sc = d_start + d_end
        if best is None or sc < best[0]:
            best = (sc, t, es, ee, d_start, d_end)
    if best is None:
        return None
    sc, t, es, ee, ds, de = best
    segs = []
    for a, b in zip(es, ee):
        lo, hi = max(a + 1, s), min(b, e)
        if lo <= hi:
            segs.append((lo, hi))
    return dict(chrom=chrom, start=s, end=e, gene=gene, strand=t.strand, tx=t["name"], segs=segs, d_start=ds, d_end=de)

comp = str.maketrans("ACGT", "TGCA")
def revcomp(sq): return sq.translate(comp)[::-1]

def circ_sequence(cid):
    info = circ_exons(cid)
    if info is None: return None, None
    if not info["segs"]:
        info["segs"] = [(info["start"], info["end"])]
    full = fetch_seq(info["chrom"], info["start"] - 1, info["end"])
    seq = "".join(full[a - info["start"]: b - info["start"] + 1] for a, b in info["segs"])
    if info["strand"] == "-": seq = revcomp(seq)
    return seq, info

fam = pd.read_csv(f"{REF}/miR_Family_Info.txt", sep="\t")
fam = fam[fam["Species ID"] == 9606].drop_duplicates("MiRBase ID").set_index("MiRBase ID")
def seed_sites(seq, mature_rna):
    """Count canonical seed sites on a CIRCULAR sequence (first 8 nt appended to catch back-splice-junction-spanning sites).
    8mer   : revcomp(nt2-8) + A          (A opposite miRNA nt1)
    7mer-m8: revcomp(nt2-8), not followed by A
    7mer-A1: revcomp(nt2-7) + A, not preceded by revcomp(nt8)   (i.e. not an 8mer)"""
    m = mature_rna.replace("U", "T")
    core8 = revcomp(m[1:8]); core7 = revcomp(m[1:7]); b8 = revcomp(m[7:8])
    L = len(seq); sc = seq + seq[:8]
    c8 = c7m8 = c7a1 = 0
    for pos in range(L):
        if sc[pos:pos + 7] == core8:
            if sc[pos + 7:pos + 8] == "A": c8 += 1
            else: c7m8 += 1
        if sc[pos:pos + 6] == core7 and sc[pos + 6:pos + 7] == "A":
            prev = sc[(pos - 1) % len(sc)] if pos > 0 else seq[-1]
            if prev != b8: c7a1 += 1
    return c8, c7m8, c7a1

def shuffled_p(seq, mature, obs_total, nshuf=500):
    chars = list(seq); cnt = 0
    for _ in range(nshuf):
        rng.shuffle(chars); s = "".join(chars)
        if sum(seed_sites(s, mature)) >= obs_total: cnt += 1
    return (cnt + 1) / (nshuf + 1)

print("\n[2] Sequence-level seed-site analysis (circRNA spliced sequence, hg38 refGene exon structure)")
seq_rows = []
for _, r in pairs.iterrows():
    seq, info = circ_sequence(r.circRNA)
    mir = r.miRNA
    if seq is None or mir not in fam.index:
        seq_rows.append(dict(circRNA=r.circRNA, miRNA=mir, note="no sequence/miRNA seq")); continue
    mature = fam.loc[mir, "Mature sequence"]
    c8, c7m8, c7a1 = seed_sites(seq, mature)
    tot = c8 + c7m8 + c7a1
    pv = shuffled_p(seq, mature, tot)
    seq_rows.append(dict(circRNA=r.circRNA, miRNA=mir, mature_seq=mature, circ_len_nt=len(seq), n_exon_segments=len(info["segs"]),
                         boundary_mismatch_nt=info["d_start"] + info["d_end"], sites_8mer=c8, sites_7mer_m8=c7m8, sites_7mer_A1=c7a1,
                         total_sites=tot, density_per_kb=1000 * tot / len(seq), p_vs_shuffled_seq=pv))
seqdf = pd.DataFrame(seq_rows)
print(seqdf.drop(columns=["mature_seq"], errors="ignore").assign(circRNA=lambda d: d.circRNA.str.replace("circ_", "")).round(3).to_string(index=False))
# background density: seed sites of ALL 6 miRNAs on the 6 circRNAs (cross-pairing) to see if the paired miRNA is special
pairs.merge(seqdf[["circRNA", "miRNA", "total_sites", "circ_len_nt"]], on=["circRNA", "miRNA"]).to_csv(f"{OUT}/results/D1_circRNA_miRNA_pairs_with_sequence.csv", index=False)
seqdf.to_csv(f"{OUT}/results/D2_seed_site_analysis.csv", index=False)

# ---------------- 3. positive control: TargetScan targets vs miRNA-mRNA correlation ----------------
print("\n[3] Positive control: are TargetScan-predicted targets negatively correlated with their miRNA in this cohort?")
ts = pd.read_csv(f"{REF}/Predicted_Targets_Info.default_predictions.txt", sep="\t", usecols=["miR Family", "Gene Symbol", "Species ID", "Seed match"])
ts = ts[(ts["Species ID"] == 9606) & ts["Seed match"].isin(["8mer", "7mer-m8"])]
fam_all = pd.read_csv(f"{REF}/miR_Family_Info.txt", sep="\t"); fam_all = fam_all[fam_all["Species ID"] == 9606]
mir2fam = dict(zip(fam_all["MiRBase ID"], fam_all["miR family"]))
fam_targets = ts.groupby("miR Family")["Gene Symbol"].apply(set).to_dict()

Mx = D["mrna"]; Mx = Mx.loc[:, ~Mx.columns.duplicated()]
genes = Mx.columns.values
res = []
good = [m for m in MIraw.columns if MIraw[m].notna().sum() >= 55 and m in mir2fam and mir2fam[m] in fam_targets]
print("   miRNAs with >=55/98 observed values and a TargetScan family:", len(good))
for m in good:
    ok = MIraw[m].notna().values
    x = rankdata(MIraw[m].values[ok]); xz = (x - x.mean()) / x.std()
    Y = rankdata(Mx.values[ok], axis=0); Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    rho = xz @ Y / ok.sum()
    tg = np.isin(genes, list(fam_targets[mir2fam[m]]))
    if tg.sum() < 50 or (~tg).sum() < 50: continue
    u = stats.mannwhitneyu(rho[tg], rho[~tg], alternative="less")
    res.append(dict(miRNA=m, family=mir2fam[m], n_targets=int(tg.sum()), median_rho_targets=np.median(rho[tg]), median_rho_nontargets=np.median(rho[~tg]),
                    delta=np.median(rho[tg]) - np.median(rho[~tg]), auc_neg=1 - u.statistic / (tg.sum() * (~tg).sum()), p_one_sided=u.pvalue))
pc = pd.DataFrame(res)
pc["fdr"] = bh(pc.p_one_sided.values)
print(f"   miRNAs evaluated: {len(pc)}; targets more negatively correlated than non-targets (delta<0): {(pc.delta < 0).sum()} ({100*(pc.delta<0).mean():.0f}%); FDR<0.05: {(pc.fdr<.05).sum()}")
print(f"   median delta = {pc.delta.median():.4f}; sign test p = {stats.binomtest(int((pc.delta<0).sum()), len(pc), .5).pvalue:.2e}; Wilcoxon p = {stats.wilcoxon(pc.delta).pvalue:.2e}")
print("   pooled AUC (target more negative than non-target), median:", round(pc.auc_neg.median(), 3))
pc.to_csv(f"{OUT}/results/D3_miRNA_target_positive_control.csv", index=False)
for m in pairs.miRNA.unique():
    row = pc[pc.miRNA == m]
    print("   ", m, "in control:", (row[["delta", "p_one_sided"]].round(4).values.tolist() if len(row) else "not evaluable (too many missing / no TargetScan family)"),
          "| observed in", int(MIraw[m].notna().sum()), "/98")
