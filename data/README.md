# Input data (not redistributed)

Place public files in the folders below, or point the environment variables to your copies. The analysis code does not modify these files.

## CPTAC glioblastoma (`data/cptac/` or `CPTAC_DATA_DIR`)

Download the cBioPortal study **gbm_cptac_2021**:

https://www.cbioportal.org/study/summary?id=gbm_cptac_2021

Required tab-separated files:

| File | Content |
|------|---------|
| `circular_rna.txt` | circRNA expression (back-splice junctions, GRCh38) |
| `mrna_seq_fpkm.txt` | mRNA expression (FPKM) |
| `protein_quantification.txt` | protein abundance (mass spectrometry) |
| `mirna_zscores.txt` | microRNA z-scores |
| `clinical_patient.txt` | clinical follow-up |
| `mutations.txt` | somatic mutations (used for IDH1 and other genes) |

The loader keeps the 98 tumours present in all four molecular layers.

## Reference files (`output/cache/ref/`)

Needed for scripts 04 and 07:

| File | Source |
|------|--------|
| `hg38ToHg19.over.chain.gz` | UCSC liftOver, GRCh38 to GRCh37 |
| `Predicted_Targets_Info.default_predictions.txt` | TargetScanHuman 7.2 |
| `miR_Family_Info.txt` | TargetScanHuman 7.2 |

Script 04 also queries the UCSC sequence API for spliced circRNA sequences.

## Independent circRNA calls (`output/cache/ext/raw/`)

Needed for scripts 07 and 09. GEO series **GSE196695** (CIRI2 calls on GRCh37):

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196695

Place the per-sample CIRI tables as gzip-compressed files named `*_circ.txt.gz` in `output/cache/ext/raw/`.

## Derived files already in this repository (`data/derived/`)

These small tables come from the first conventional workflow and are used only for comparison:

- `Supp.16_Cluster_assignments_cluster_assignments.csv`
- `Supp.20_Biomarker_signature_scores_biomarker_signature_scores.csv`
- `Supp.9_Bootstrap_stability_results_bootstrap_stability_results.csv`
