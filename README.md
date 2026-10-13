# Analysis code for circular RNA associations in glioblastoma

This repository holds the analysis code for the research article:

> Dogan S, Schnauß J, Leopold M, Kirsten T. Host gene expression and detection depth explain most circular RNA associations in a glioblastoma multi-omic cohort. Manuscript submitted to *BMC Genomic Data*.

The scripts repeat a conventional circRNA co-expression workflow on the CPTAC glioblastoma cohort and then test whether hubs, clusters, microRNA-sponge pairs and survival associations survive depth adjustment, unfiltered screens, sequence checks and an independent dataset (GSE196695).

Corresponding author: Senol Dogan ([Senol.Dogan@medizin.uni-leipzig.de](mailto:Senol.Dogan@medizin.uni-leipzig.de)), Leipzig University.

Repository: [https://github.com/ul-mds/circular-RNA-in-Glioblastoma](https://github.com/ul-mds/circular-RNA-in-Glioblastoma)

## Contents

| Path | Description |
|------|-------------|
| `scripts/` | Analysis scripts, numbered in the order they should be run |
| `results/` | Derived tables used in the article (Tables S1 to S28) |
| `results_logs/` | Text logs from the runs that produced those tables |
| `data/derived/` | Small cluster and signature files needed to compare with the first workflow |
| `data/cptac/` | Empty placeholder for the public CPTAC files (not redistributed) |
| `output/` | Created when the scripts are re-run |
| `LICENSE` | MIT licence |
| `requirements.txt` | Python package versions used for the article |

Raw CPTAC, GEO, TargetScan and genome files are public and are not stored here. See [`data/README.md`](data/README.md).

## Requirements

- Python 3.13 (tested on macOS with Python 3.13.3)
- Packages in `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## How to run

1. Download the CPTAC files listed in `data/README.md` into `data/cptac/` (or set `CPTAC_DATA_DIR`).
2. For scripts 04, 07, 09 and 10, add the reference and GEO files described in `data/README.md`.
3. From the repository root:

```bash
export CPTAC_DATA_DIR="./data/cptac"
export ANALYSIS_OUT="./output"
python3 scripts/01_data_provenance.py
python3 scripts/02_hub_reanalysis.py
python3 scripts/03_subtype_reanalysis.py
python3 scripts/04_cerna_validation.py
python3 scripts/05_robustness.py
python3 scripts/06_survival.py
python3 scripts/07_external_circRNA_GSE196695.py
python3 scripts/08_pathology_sensitivity.py
python3 scripts/09_make_figures.py
python3 scripts/10_supp_figures.py
```

Or: `bash run_all.sh`

Random seeds are fixed in each script that uses random numbers. Re-running writes tables to `output/results/` and figures to `output/figures/`. The `results/` folder is the set used in the article.

## Mapping to the article

| Script | Role in the article |
|--------|---------------------|
| `01_data_provenance.py` | Sample counts, missing-value filter, detection depth (Table 1, Table S1, Table S2) |
| `02_hub_reanalysis.py` | Correlation screen, network hubs, permutation null, bootstrap ranks (Table 2, Fig. 2) |
| `03_subtype_reanalysis.py` | Clustering, Random Forest, consensus PAC (Table 3, Fig. 3) |
| `04_cerna_validation.py` | circRNA–microRNA screen and seed sites (Table 4, Fig. 5) |
| `05_robustness.py` | Imputation, unfiltered screen, depth adjustment, host-gene correlations (Fig. 4) |
| `06_survival.py` | Cox models and Kaplan–Meier curves (Table 6, Fig. 7) |
| `07_external_circRNA_GSE196695.py` | Junction replication in GSE196695 (Table 5, Fig. 6) |
| `08_pathology_sensitivity.py` | Repeat of key results without tumours that failed pathology review (Table S28) |
| `09_make_figures.py` | Main figures (TIFF, 300 dpi, RGB) |
| `10_supp_figures.py` | Figs. S1 to S3 |

## Data sources

- CPTAC glioblastoma multi-omic data: cBioPortal study [gbm_cptac_2021](https://www.cbioportal.org/study/summary?id=gbm_cptac_2021) ([Wang et al., *Cancer Cell*, 2021](https://doi.org/10.1016/j.ccell.2021.01.006)).
- Independent circRNA calls: GEO [GSE196695](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196695).
- microRNA target predictions: [TargetScanHuman 7.2](https://www.targetscan.org/vert_72/).
- Genome coordinates: UCSC Genome Browser (GRCh38 / hg19 chain).

## Licence

MIT Licence. See `LICENSE`. The public datasets remain under the terms of their original repositories.

## Citation

Please cite the article when using this code. The same scripts and logs are included in Supplementary Code with the manuscript.
