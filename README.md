# Gene Expression Coding Exercise

Analysis of Black and White ovarian cancer gene expression data: filtering, PCA, subtype proportions, and supervised classification of `ClusterK4_kmeans` subtypes.

Full assignment details are in [`docs/instructions.md`](docs/instructions.md).

## Requirements

- Python **3.10**
- Packages listed in [`requirements.txt`](requirements.txt):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Installation

Requires [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). From the project root:

```bash
micromamba create -n gene_expr -y python=3.10
micromamba activate gene_expr
pip install -r requirements.txt
```

## Data

Data and instructions are available on [Google Drive](https://drive.google.com/drive/folders/11EEWzD87G6qnyJHXqRaSKFv0Jgla8Ra2). Download the `data/` folder and place it in the project root (not tracked in git):

```
data/
  supp_table_1_GlobalMAD_genelist.csv
  supp_table_2_CommonGenes_genelist.csv
  supp_table_3_main_black_metadata_table.tsv
  supp_table_4_main_white_metadata_table.tsv
  supp_table_6_black_expr.tsv
  supp_table_7_white_expr.tsv
```

## Project layout

| Path | Contents |
|------|----------|
| `src/` | Numbered scripts for each exercise step |
| `out/` | Filtered tables and merged metadata |
| `figs/` | Plots  |
| `models/` | Saved classifiers (`.pkl`) |
| `docs/` | Assignment instructions and written answers |

## Running the pipeline

Run scripts from the project root in order. Each script accepts `-h` for full options.

**Task 1 — Filter expression to GlobalMAD genes**
Writes filtering summary to output folder

```bash
python src/1_filter_genelist.py \
  -meta data/supp_table_3_main_black_metadata_table.tsv \
  -expr data/supp_table_6_black_expr.tsv \
  -genes data/supp_table_1_GlobalMAD_genelist.csv \
  -sample black -gene_list_name GlobalMAD

python src/1_filter_genelist.py \
  -meta data/supp_table_4_main_white_metadata_table.tsv \
  -expr data/supp_table_7_white_expr.tsv \
  -genes data/supp_table_1_GlobalMAD_genelist.csv \
  -sample white -gene_list_name GlobalMAD

python src/1_plot_filtered_data.py \
  -summary out/filter_summary.tsv
```

**Task 2 — Count shared genes between gene lists**

```bash
python src/2_shared_genes.py \
  -list1 data/supp_table_1_GlobalMAD_genelist.csv \
  -list2 data/supp_table_2_CommonGenes_genelist.csv
```

**Task 3 — PCA on combined Black and White expression**

```bash
python src/3_dim_reduction.py \
  -black_expr out/black_GlobalMAD_filtered.tsv \
  -black_meta out/black_filtered_metadata.tsv \
  -white_expr out/white_GlobalMAD_filtered.tsv \
  -white_meta out/white_filtered_metadata.tsv
```

**Task 6 — Subtype proportions by race**

```bash
python src/6_proportions.py -meta out/black_white_meta.tsv
```

**Task 7 — Train classifier on White individuals**

```bash
python src/7_classifier.py \
  -expr out/white_GlobalMAD_filtered.tsv \
  -meta out/white_filtered_metadata.tsv \
  -sample white
```

**Task 8 — Apply White-trained model to Black individuals**

```bash
python src/8_predict_black_ind.py \
  -expr out/black_GlobalMAD_filtered.tsv \
  -meta out/black_filtered_metadata.tsv \
  -model models/white_logreg.pkl \
  -sample black
```

**Task 9 — Train on combined data with 20% holdout**

```bash
python src/9_joint_classifier.py \
  -expr out/black_white_expr.tsv \
  -meta out/black_white_meta.tsv \
  -sample combined
```

## Written answers

Task prompts are in [`docs/instructions.md`](docs/instructions.md). Written commentary for specific tasks:

| Task | Document |
|------|----------|
| 4 — PCA interpretation | [`docs/4_pca_interpritation.md`](docs/4_pca_interpritation.md) |
| 5 — Additional filtering ideas | [`docs/5_pca_additional_filtering_ideas.md`](docs/5_pca_additional_filtering_ideas.md) |
| 7 — White classifier performance | [`docs/7_classifier_performance_white.md`](docs/7_classifier_performance_white.md) |
| 8 — Black transfer performance | [`docs/8_classifier_performance_black.md`](docs/8_classifier_performance_black.md) |

An exploratory notebook is  available at [`src/EDA.ipynb`](src/EDA.ipynb).
