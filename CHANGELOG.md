# Changelog

All notable changes to RepoGen are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-21

First public release. RepoGen identifies drug repurposing candidates from GWAS
summary statistics along three independent lines of evidence.

### Added

**Branch A, gene and drug-target enrichment**
- MAGMA gene-level and gene-set analysis, with proximity or H-MAGMA
  chromatin-interaction annotation
- Drug-target enrichment against ChEMBL, optionally augmented with PDSP Ki and
  DGIdb
- ATC drug-class enrichment with a Wilcoxon concordance check and an optional
  drug-label permutation null
- Two-tier gene-count thresholds, so class-level tests can draw on a more
  inclusive drug pool than individual-drug headlines

**Branch B, expression signature reversal**
- S-PrediXcan imputed expression association across 13 brain tissues
- LINCS L1000 drug signature extraction with cell-line-aware consensus
  weighting
- Negative-correlation scoring with per-tissue FDR and a genomic-control
  calibration sidecar

**Branch C, Mendelian randomisation**
- cis-eQTL MR against eQTLGen and MetaBrain
- Colocalisation with configurable priors, plus Steiger directionality
  filtering
- Weak-instrument filtering on the F statistic

**Cross-branch**
- Convergence analysis reporting drugs supported by more than one branch
- Publication-ready figures as PNG and PDF
- Self-contained HTML report, plus CSV, JSON, and Excel exports
- JSON metadata sidecars on every analysis, recording parameters, input
  checksums, and external tool versions

**Deployment**
- Snakemake workflow covering all branches, with per-rule CPU, memory, and
  runtime budgets
- SLURM submission through Snakemake profiles; `profiles/create` targets
  King's College London CREATE and is documented for adaptation to other sites
- `repogen setup-resources` downloads and verifies external data, with
  `--branch` to fetch only what a given branch needs
- `repogen validate` checks configuration and confirms required resources are
  present before a run starts
- Pinned conda and pip lock files for reproducible environments
- Container image built and published by CI on tagged releases

### Validated

All three branches were run end to end on a SLURM cluster (KCL CREATE) against
PGC3 schizophrenia wave 3, 7.66M variants.

Per-rule CPU, memory and runtime budgets were set from those runs and are
documented in `workflows/rules/common.smk`.

### Notes

- MAGMA is not redistributed. Its licence permits free academic use but
  forbids redistribution, so `setup-resources` fetches it from the authors'
  site. It is absent from both the conda environment and the container image.
- 13 of the 32 catalogued resources require manual acquisition because they sit
  behind registration or data-use agreements. `setup-resources` reports exactly
  which, and where to get them.
- macOS is not supported; several external tools ship as Linux binaries only.

[1.0.0]: https://github.com/IvanAndru/RepoGen/releases/tag/v1.0.0
