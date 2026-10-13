"""
A. Data provenance / transparency numbers (for Methods) + sample accounting
"""
import numpy as np, pandas as pd
from common import *

D = load(force=True)
rows = []
print("Sample sets in raw files:", D["sample_sets"], " common (incl. clinical):", D["common_initial"], " final:", len(D["samples"]))
for k, label in [("circ", "circRNA"), ("mrna", "mRNA"), ("prot", "protein"), ("mirna", "miRNA")]:
    raw = D[f"{k}_raw"]
    rows.append(dict(layer=label, rows_in_file=D[f"{k}_file_rows"],
                     features_before_missing_filter=D[f"{k}_n_before"],
                     features_after_missing_filter_lt50pct=D[f"{k}_n_after"],
                     removed=D[f"{k}_n_before"] - D[f"{k}_n_after"],
                     pct_cells_missing_in_retained_features=100 * D[f"{k}_frac_imputed_after_filter"],
                     pct_features_with_any_missing=100 * raw.isna().any().mean(),
                     pct_features_with_gt20pct_missing=100 * (raw.isna().mean() > .2).mean()))
t = pd.DataFrame(rows)
print(t.round(2).to_string(index=False))
t.to_csv(f"{OUT}/results/A1_data_accounting.csv", index=False)

# circRNA zero inflation / detection
C = D["circ"]
det_prev = (C > 0).mean()
print("\ncircRNA matrix:", C.shape, " zero fraction: %.3f" % (C == 0).values.mean())
print("detection prevalence (fraction of samples with BSJ>0) quantiles:\n", det_prev.quantile([.05, .25, .5, .75, .95]).round(3).to_dict())
for thr in [1, 5, 10, 20, 30, 50]:
    print(f"  circRNAs detected in >= {thr}% samples: {(det_prev >= thr/100).sum()}")
per_sample = pd.DataFrame({"n_detected_circ": (C > 0).sum(1), "total_circ_signal": C.sum(1)})
print(per_sample.describe().round(1))
per_sample.to_csv(f"{OUT}/results/A2_per_sample_circ_QC.csv")

# Check names/genome build clues
ann = D["circ_ann"]
print("\ncirc id examples:", list(ann.index[:3]))
print("DESCRIPTION counts (biotype):", ann["DESCRIPTION"].value_counts().head(6).to_dict())
mut = pd.read_csv(f"{ROOT}/mutations.txt", sep="\t", usecols=["Center", "NCBI_Build", "Tumor_Sample_Barcode"], low_memory=False)
print("mutation file build/center:", mut["NCBI_Build"].unique(), mut["Center"].unique()[:5], " n tumors:", mut["Tumor_Sample_Barcode"].nunique())
ids = pd.Series(D["samples"])
print("sample id prefixes:", ids.str[:3].value_counts().to_dict(), " (C3L/C3N = CPTAC-3 case IDs)")
# which sample is dropped relative to 99 patients?
clin = D["clinical"]
print("patients in clinical:", len(clin), " not in final:", sorted(set(clin.index) - set(D["samples"])))
