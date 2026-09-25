# Changelog

All notable changes to RepoGen are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Branch C runs faster with identical results. The GWAS is indexed by rsID
  and by position once per run instead of being copied for every gene, and
  PLINK clumping reads a per-chromosome copy of the reference panel, written
  beside it on first use with allele order kept.
- Branch C chooses instruments among SNPs the GWAS can use. Candidates are
  matched to the GWAS before LD clumping, so a strong eQTL SNP missing from
  the GWAS no longer clumps away its usable neighbours; the lead instrument
  and the clumping order follow the eQTL |z| rather than eQTLGen's floored P
  values, and where clumping is needed only SNPs in the LD reference panel
  are kept. Palindromic SNPs are kept when allele frequencies in the eQTL data
  and the GWAS confirm their strand, and a position join is attempted only
  between data on the same genome build. Results change.

### Added

- `prepare_data/gwas_mr.parquet`, the GWAS preparation Branch C reads: the
  same QC as `gwas_standardized.parquet` without reference-panel
  harmonisation, which removes every palindromic SNP.
- The setup-resources entry `eqtlgen_allele_frequency` and the config key
  `mr.eqtl_sources[].allele_frequency_path`: eQTLGen's own allele
  frequencies, used for its z-to-beta conversion and the palindromic check.
- Per-gene result columns describing the instrument set: the lead
  instrument, whether it is palindromic or outside the LD panel, candidate
  and usable SNP counts, palindromic SNPs kept and dropped, and the distance
  from the gene to the nearest instrument.

### Fixed

- Branch C read whichever `*.txt*` or `*.tsv*` file the eQTLGen directory
  listed first. It now ignores README and checksum files and stops with an
  error if more than one table remains.
- A Branch C run with no drug matches left an earlier run's
  `mr_drug_matches.csv` in place beside an empty parquet; the CSV is now
  always rewritten.

## [1.0.1] - 2026-08-04

### Fixed

- The documented container pull command used the wrong tag. The git tag is
  `v1.0.0`, but the published image tags drop the leading `v`, so
  `docker://ghcr.io/ivanandru/repogen:v1.0.0` does not resolve. Use `1.0.0`.
- Tests that read repository files resolved them against the working
  directory, so the suite only passed when run from a checkout root. They now
  resolve against the test file's location and pass from anywhere, including
  from a writable directory alongside a read-only container image.

### Added

- Installation notes for running the image on a cluster: bind the filesystem
  a symlinked scratch path points at, not just the symlink, and run from a
  writable directory because Apptainer mounts images read-only.

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

[1.0.1]: https://github.com/IvanAndru/RepoGen/releases/tag/v1.0.1
[1.0.0]: https://github.com/IvanAndru/RepoGen/releases/tag/v1.0.0
