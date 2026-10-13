"""
Shared data loader for the glioblastoma circRNA re-analysis.

Public CPTAC files are read from CPTAC_DATA_DIR (or GBM_ROOT). Derived tables
are written to ANALYSIS_OUT/results. Nothing in this module changes the
manuscript files.

Expected CPTAC files (tab-separated, cBioPortal study gbm_cptac_2021):
  circular_rna.txt, mrna_seq_fpkm.txt, protein_quantification.txt,
  mirna_zscores.txt, clinical_patient.txt, mutations.txt
"""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CPTAC_DATA_DIR", os.environ.get("GBM_ROOT", REPO_ROOT / "data" / "cptac")))
OUT = str(Path(os.environ.get("ANALYSIS_OUT", REPO_ROOT / "output")))
ROOT = str(DATA_DIR)
DERIVED = str(REPO_ROOT / "data" / "derived")
CACHE = os.path.join(OUT, "cache", "data.pkl")
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)


def _sample_cols(df):
    return [c for c in df.columns if c.startswith("C3")]


def _prep(df, samples, id_col, impute=True, missing_thr=0.5):
    """Filter features with 50% or more missing values and optionally impute the rest with the column median."""
    d = df.set_index(id_col)
    avail = [s for s in samples if s in d.columns]
    sub = d[avail].T.apply(pd.to_numeric, errors="coerce")
    n_before = sub.shape[1]
    sub = sub.loc[:, sub.isna().mean() < missing_thr]
    n_after = sub.shape[1]
    raw = sub.copy()
    frac_imputed = float(raw.isna().values.mean())
    if impute:
        sub = sub.fillna(sub.median())
    return sub, raw, n_before, n_after, frac_imputed


def load(force=False):
    if os.path.exists(CACHE) and not force:
        return pickle.load(open(CACHE, "rb"))
    required = [
        "circular_rna.txt",
        "mrna_seq_fpkm.txt",
        "protein_quantification.txt",
        "mirna_zscores.txt",
        "clinical_patient.txt",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        raise FileNotFoundError(
            "CPTAC files not found in %s: %s. "
            "Download the gbm_cptac_2021 study from cBioPortal and set CPTAC_DATA_DIR. "
            "See data/README.md." % (ROOT, ", ".join(missing))
        )
    circ = pd.read_csv(os.path.join(ROOT, "circular_rna.txt"), sep="\t")
    mrna = pd.read_csv(os.path.join(ROOT, "mrna_seq_fpkm.txt"), sep="\t")
    prot = pd.read_csv(os.path.join(ROOT, "protein_quantification.txt"), sep="\t")
    mirna = pd.read_csv(os.path.join(ROOT, "mirna_zscores.txt"), sep="\t")
    clin = pd.read_csv(os.path.join(ROOT, "clinical_patient.txt"), sep="\t", skiprows=4)
    clin.columns = clin.columns.str.strip()
    clin = clin[clin["PATIENT_ID"].notna()]

    sets = dict(
        circ=set(_sample_cols(circ)),
        mrna=set(_sample_cols(mrna)),
        prot=set(_sample_cols(prot)),
        mirna=set(_sample_cols(mirna)),
        clin=set(clin["PATIENT_ID"]),
    )
    common = sorted(set.intersection(*sets.values()))

    out = {"sample_sets": {k: len(v) for k, v in sets.items()}, "common_initial": len(common)}
    mats = {}
    for name, df, idc in [
        ("circ", circ, "ENTITY_STABLE_ID"),
        ("mrna", mrna, "Hugo_Symbol"),
        ("prot", prot, "Composite.Element.REF"),
        ("mirna", mirna, "Hugo_Symbol"),
    ]:
        m, raw, nb, na, fi = _prep(df, common, idc)
        mats[name] = m
        out[f"{name}_raw"] = raw
        out[f"{name}_n_before"] = nb
        out[f"{name}_n_after"] = na
        out[f"{name}_frac_imputed_after_filter"] = fi
        out[f"{name}_file_rows"] = df.shape[0]
    final = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
    for k in mats:
        mats[k] = mats[k].loc[final]
        out[f"{k}_raw"] = out[f"{k}_raw"].loc[final]
    out["samples"] = final
    out.update(mats)
    ann = circ[["ENTITY_STABLE_ID", "NAME", "DESCRIPTION", "GENE_SYMBOL"]].set_index("ENTITY_STABLE_ID")
    out["circ_ann"] = ann
    out["clinical"] = clin.set_index("PATIENT_ID")
    pickle.dump(out, open(CACHE, "wb"))
    return out


def rank_cols(df):
    """Column-wise average ranks (Spearman via Pearson on ranks)."""
    return df.rank(axis=0, method="average")


def spearman_matrix(X, Y):
    """X: n x p, Y: n x q arrays. Returns a p x q Spearman correlation matrix. Input must not contain NaN."""
    from scipy.stats import rankdata
    Rx = np.apply_along_axis(rankdata, 0, X)
    Ry = np.apply_along_axis(rankdata, 0, Y)
    Rx = (Rx - Rx.mean(0)) / (Rx.std(0) + 1e-12)
    Ry = (Ry - Ry.mean(0)) / (Ry.std(0) + 1e-12)
    return Rx.T @ Ry / X.shape[0]


def rho_to_p(r, n):
    from scipy import stats
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), n - 2)


def bh(p):
    from statsmodels.stats.multitest import multipletests
    return multipletests(p, method="fdr_bh")[1]
