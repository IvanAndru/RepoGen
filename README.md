# RepoGen

Genomic-driven drug repurposing from GWAS summary statistics.

RepoGen takes a genome-wide association study for a disease and asks which
existing drugs are implicated by that genetic signal using several methodologies.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/snakemake-%E2%89%A58.0-brightgreen.svg)](https://snakemake.github.io)

## The three branches

| Branch | Question | Method |
|---|---|---|
| **A** | Are risk loci enriched among the targets of particular drugs or drug classes? | MAGMA gene and gene-set analysis, then drug-target and ATC-class enrichment |
| **B** | Does a drug reverse the disease's predicted expression signature? | S-PrediXcan imputed expression, correlated against LINCS L1000 drug profiles |
| **C** | Is a drug target causally linked to the disease? | Mendelian randomisation on cis-eQTLs, with colocalisation |

Each branch runs independently, so you can use one, two, or all three. A final
convergence step reports drugs supported by more than one line of evidence.

## Install

```bash
git clone https://github.com/IvanAndru/RepoGen.git
cd RepoGen
conda env create -f envs/repogen.yaml
conda activate repogen
pip install -e .
```

RepoGen needs a Linux environment (native Linux, WSL, or an HPC login node).
For an exactly reproducible environment use the pinned lock file instead of
`envs/repogen.yaml`; see [docs/installation.md](docs/installation.md).

## Get the reference data

```bash
repogen setup-resources --branch a
```

This downloads what it can and tells you what you must fetch yourself. Of the
32 catalogued resources, 13 require manual acquisition because they sit behind
registration or data-use agreements. Restrict downloads to the branches you
plan to run:

| `--branch` | Resources | Notes |
|---|---|---|
| `a` | 9 (+10 shared) | MAGMA, 1000 Genomes LD panel, MSigDB, ChEMBL |
| `b` | 10 (+10 shared) | PredictDB models, LINCS L1000 (~33 GB) |
| `c` | 5 (+10 shared) | eQTLGen and/or MetaBrain cis-eQTLs |

Full disk footprint exceeds 100 GB with all branches.
See [docs/data_requirements.md](docs/data_requirements.md).

## Configure and check

Edit `configs/config.yaml` to point at your GWAS and set study metadata, then:

```bash
repogen validate --config configs/config.yaml
```

`validate` checks the configuration *and* verifies that every resource the
selected branches need is actually present, so you find missing data before a
long run rather than during one. See [docs/configuration.md](docs/configuration.md).

## Run

```bash
repogen run --config configs/config.yaml --step branch_a --cores 8
```

Preview without executing using `--dryrun`. Arguments after `--` pass straight
through to Snakemake:

```bash
repogen run --config configs/config.yaml -- --rerun-incomplete
```

### Choosing `--step`

| `--step` | Runs |
|---|---|
| `branch_a` | MAGMA, pathway, drug and ATC enrichment |
| `branch_b` | S-PrediXcan and signature reversal |
| `branch_c` | Mendelian randomisation |
| `magma` / `drug` | Just that part of Branch A |
| `correlation` / `mr` | Just Branch B / Branch C |
| `available` | A report from whichever branches have already produced results |
| `all` (default), `full` | All three branches, then the combined report |

Important note: **`all` requires all three branches to run**. If
you only have data for one or two, use `available` - it builds the combined
report from whatever exists rather than failing on missing inputs. Any rule
name from the workflow also works as a `--step`.

### On a cluster

RepoGen submits each rule as its own SLURM job via a Snakemake profile:

```bash
repogen run --config configs/config.yaml --step branch_a --profile profiles/create
```

`profiles/create` targets King's College London CREATE. To adapt it to another
site, copy the directory and change `slurm_partition` (and `slurm_account` if
your scheduler requires one). Nothing else in it is site-specific: per-rule CPU,
memory, and runtime budgets live with the workflow rules and travel with the
pipeline.

## Output

Results land under `output_dir/<study_name>/`:

```
prepare_data/              standardised GWAS, shared by all branches
gene_annotation/           gene ID mapping and annotation
magma/                     gene and gene-set association results  (A)
drug_enrichment/           per-drug enrichment statistics         (A)
atc_enrichment/            drug-class enrichment                  (A)
spredixcan/                imputed expression associations        (B)
negative_correlation/      drug-signature reversal scores         (B)
mr/                        causal estimates and colocalisation    (C)
  sensitivity/mhc_excluded/  the same analysis with the MHC dropped
plots/                     publication-ready figures (PNG + PDF)
  _status/                   one JSON marker per figure
exports/                   CSV, JSON, and Excel tables, by branch
report/full/report.html    combined report (needs all three branches)
report/available/report.html  report from whichever branches ran
logs/                      per-rule logs
```

The Mendelian randomisation directory is
`mr/`, not `mendelian_randomisation/`, and the report is under `report/`
(singular) in either a `full/` or `available/` subdirectory depending on the
target you built.

Optional figures write a JSON marker to `plots/_status/` rather than failing
when they cannot be drawn. If a figure is missing, that marker records whether
it was skipped for an empty table, a missing column, or too few branches, so a
blank space in the report is always explained.

Every analysis writes a JSON metadata sidecar recording parameters, input
checksums, and the versions of external tools used, so a result can be traced
back to the exact conditions that produced it.

## Documentation

- [Installation](docs/installation.md) - environments, lock files, containers
- [Configuration](docs/configuration.md) - every config option explained
- [Data requirements](docs/data_requirements.md) - acquiring all external data

## Requirements

Python 3.11+, Snakemake 8+, and two external tools: MAGMA and PLINK 1.9.
MAGMA cannot be redistributed under its licence, so `setup-resources` fetches
it from the authors' site.

S-PrediXcan is implemented natively rather than shelled out to MetaXcan, so
Branch B needs only the PredictDB model files.

## Citing

If RepoGen contributes to published work, please cite it using the metadata in
[CITATION.cff](CITATION.cff). Please also cite the underlying methods and
databases you used: MAGMA (de Leeuw et al. 2015), S-PrediXcan (Barbeira et al.
2018), coloc (Giambartolomei et al. 2014), ChEMBL, LINCS L1000, PredictDB, and
whichever eQTL resource you ran Branch C against.

## Licence

MIT - see [LICENSE](LICENSE). The licence covers RepoGen's own source code.
External tools and datasets carry their own terms, summarised in `LICENSE` and
detailed in [docs/data_requirements.md](docs/data_requirements.md).
