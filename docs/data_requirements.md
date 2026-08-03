# RepoGen Data and Resource Installation Guide

This guide explains how to install all RepoGen resources in the correct order so the full pipeline can run.

It covers:
- automatic downloads handled by `repogen setup-resources`
- manual resources that require user action
- derived resources that are generated after manual files are present
- both local (Linux/WSL) and HPC workflows

This document matches the current `configs/resources.yaml` and current code paths in `repogen/data/resources.py`.

## 1) Before you start

You need:
- a Linux shell (native Linux, WSL, or HPC login node)
- the RepoGen repository cloned locally
- enough disk space for large resources

Approximate disk footprint for all resources can exceed 100 GB depending on GTEx and PredictDB selections.

Recommended minimum free space:
- branch A/B only: 40 to 60 GB
- full resources including GTEx and MetaBrain: 120+ GB

## 2) Use a consistent repo root

All commands below assume you are in the repository root, for example:
- local or WSL: `~/RepoGen`
- HPC: `/path/to/RepoGen`

Set:

```bash
cd /path/to/RepoGen
```

## 3) Prepare the environment

Create and activate the environment:

```bash
conda env create -f envs/repogen.yaml
conda activate repogen
pip install -e .
```

Check CLI:

```bash
repogen info
```

## 4) Run automatic resource setup first

Run:

```bash
repogen setup-resources --config configs/resources.yaml --target-dir .
```

What this does:
- downloads automatic resources
- skips manual resources and prints where to place them
- generates derived resources only if source files already exist

Important:
- this command is idempotent
- rerunning is safe and is the standard way to re-check status

## 5) Install manual resources (in order)

Install manual resources in the order below, then rerun `setup-resources`.

### 5.1 `ncbi_gene_loc` (required for MAGMA proximity annotation)

Target path:
- `resources/reference/NCBI37.3.gene.loc`

Expected format:
- MAGMA gene location text file (tab-delimited)
- no compression at final path

Source:
- MAGMA website: <https://cncr.nl/research/magma/>
- historical direct URL used in many MAGMA tutorials: `https://ctg.cncr.nl/software/MAGMA/aux_files/NCBI37.3.zip`

Notes:
- the historical direct URL may redirect depending on server behavior
- if it redirects, download from the MAGMA page manually

Example command sequence:

```bash
mkdir -p resources/reference /tmp/repogen_dl
curl -L "https://ctg.cncr.nl/software/MAGMA/aux_files/NCBI37.3.zip" -o /tmp/repogen_dl/NCBI37.3.zip
file /tmp/repogen_dl/NCBI37.3.zip
unzip -j /tmp/repogen_dl/NCBI37.3.zip "NCBI37.3.gene.loc" -d resources/reference/
```

The `file` command must report a zip archive, not HTML text.

Validation:

```bash
test -s resources/reference/NCBI37.3.gene.loc && head -n 3 resources/reference/NCBI37.3.gene.loc
```

### 5.2 `biomart_dico1`, `biomart_dico2`, `biomart_dico3` (required for gene ID conversion)

Target paths:
- `resources/reference/biomart_dico1`
- `resources/reference/biomart_dico2`
- `resources/reference/biomart_dico3`

Source:
- Ensembl BioMart: <https://www.ensembl.org/biomart/martview>

Dataset and build:
- database: Ensembl Genes
- dataset: Human genes (GRCh37 archive for hg19 compatibility)

Export each file as plain TSV with exactly 2 columns:

`biomart_dico1`:
- column 1: Ensembl Gene ID
- column 2: Gene symbol

`biomart_dico2`:
- column 1: Gene symbol
- column 2: Ensembl Gene ID

`biomart_dico3`:
- column 1: UniProtKB/Swiss-Prot ID
- column 2: Ensembl Gene ID

Critical formatting rule:
- remove any header row
- files must be two-column TSV text files

Validation:

```bash
awk -F'\t' 'NF<2{bad=1} END{exit bad}' resources/reference/biomart_dico1
awk -F'\t' 'NF<2{bad=1} END{exit bad}' resources/reference/biomart_dico2
awk -F'\t' 'NF<2{bad=1} END{exit bad}' resources/reference/biomart_dico3
head -n 3 resources/reference/biomart_dico1
```

### 5.3 `pdsp_ki` (optional source in drug loader)

Target path:
- `resources/drugs/pdsp_ki.csv`

Source:
- PDSP Ki database page: <https://pdsp.unc.edu/databases/kidb.php>

Expected format:
- CSV
- include drug and target columns (loader auto-detects common names)

Validation:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("resources/drugs/pdsp_ki.csv", nrows=5)
print(df.columns.tolist())
print(df.head(3).to_string(index=False))
PY
```

### 5.4 LINCS manual resources (`lincs_l1000`, `lincs_gene_metadata`, `lincs_compound_metadata`)

Target paths:
- `resources/drug_signatures/level5_beta_trt_cp_n720216x12328.gctx`
- `resources/drug_signatures/geneinfo_beta.txt`
- `resources/drug_signatures/compoundinfo_beta.txt`

Source:
- CMap 2020 page: <https://clue.io/data/CMap2020>

Expected formats:
- `.gctx` for L1000 matrix
- tab-delimited `.txt` metadata files

Important:
- use these exact filenames at these exact paths
- `lincs_gene_info.tsv` is derived from `geneinfo_beta.txt` during `setup-resources`

Validation:

```bash
ls -lh resources/drug_signatures/level5_beta_trt_cp_n720216x12328.gctx
ls -lh resources/drug_signatures/geneinfo_beta.txt
ls -lh resources/drug_signatures/compoundinfo_beta.txt
head -n 1 resources/drug_signatures/geneinfo_beta.txt
head -n 1 resources/drug_signatures/compoundinfo_beta.txt
```

### 5.5 PredictDB resources (`predixcan_models`, `predixcan_covariances`)

Target directories:
- `resources/predixcan_models/`
- `resources/predixcan_models/covariances/`

Source:
- PredictDB portal: <https://predictdb.org/>

RepoGen filename contract for model type `mashr`:
- model DB: `mashr_<Tissue>.db`
- covariance: `mashr_<Tissue>.txt.gz`

Default brain tissue preset (`brain_13`) requires these tissues:
- `Brain_Amygdala`
- `Brain_Anterior_cingulate_cortex_BA24`
- `Brain_Caudate_basal_ganglia`
- `Brain_Cerebellar_Hemisphere`
- `Brain_Cerebellum`
- `Brain_Cortex`
- `Brain_Frontal_Cortex_BA9`
- `Brain_Hippocampus`
- `Brain_Hypothalamus`
- `Brain_Nucleus_accumbens_basal_ganglia`
- `Brain_Putamen_basal_ganglia`
- `Brain_Spinal_cord_cervical_c-1`
- `Brain_Substantia_nigra`

Validation:

```bash
tissues=(
  Brain_Amygdala
  Brain_Anterior_cingulate_cortex_BA24
  Brain_Caudate_basal_ganglia
  Brain_Cerebellar_Hemisphere
  Brain_Cerebellum
  Brain_Cortex
  Brain_Frontal_Cortex_BA9
  Brain_Hippocampus
  Brain_Hypothalamus
  Brain_Nucleus_accumbens_basal_ganglia
  Brain_Putamen_basal_ganglia
  Brain_Spinal_cord_cervical_c-1
  Brain_Substantia_nigra
)
for t in "${tissues[@]}"; do
  test -f "resources/predixcan_models/mashr_${t}.db" || echo "Missing model: $t"
  test -f "resources/predixcan_models/covariances/mashr_${t}.txt.gz" || echo "Missing covariance: $t"
done
```

### 5.6 `eqtlgen` (required for default MR source)

Target directory:
- `resources/eqtl/eqtlgen/`

Source:
- eQTLGen landing page: <https://www.eqtlgen.org/cis-eqtls.html>

Expected format:
- tab-delimited text (optionally gzipped)
- must contain required columns:
  - `Pvalue`
  - `SNP`
  - `SNPChr`
  - `SNPPos`
  - `AssessedAllele`
  - `OtherAllele`
  - `Zscore`
  - `Gene`
  - `NrSamples`

Recommended:
- keep one main cis-eQTL file in this folder to avoid loading the wrong file by accident

Validation:

```bash
zcat resources/eqtl/eqtlgen/*.gz | head -n 1
```

If uncompressed:

```bash
head -n 1 resources/eqtl/eqtlgen/*.txt
```

### 5.7 `metabrain_cortex` (raw files) and derived `metabrain_cortex_normalized`

Target directory for raw files:
- `resources/eqtl/metabrain/`

Source:
- MetaBrain files page: <https://download.metabrain.nl/files.html>

Expected raw files:
- 22 chromosome files matching pattern `*cortex*chr*.txt.gz`
- keep original filenames (for reliable pattern matching)

Derived output generated by setup:
- `resources/eqtl/metabrain/metabrain_cortex_normalized.tsv.gz`

Validation:

```bash
ls resources/eqtl/metabrain/*cortex*chr*.txt.gz | wc -l
```

Expected:
- `22`

### 5.8 `gtex_v8_eqtl` (manual, future MR use)

Target directory:
- `resources/eqtl/gtex_v8/`

Source:
- GTEx datasets page: <https://gtexportal.org/home/datasets>

Access:
- some files require controlled-access authorization (dbGaP)

Current note:
- this is kept as a manual resource in `resources.yaml`
- use when you explicitly configure GTEx as an MR source

## 6) Regenerate derived resources and re-check status

If a derived file was generated before a code fix, delete it first so it is rebuilt cleanly:

```bash
rm -f resources/drug_signatures/lincs_gene_info.tsv
rm -f resources/eqtl/metabrain/metabrain_cortex_normalized.tsv.gz
```

After placing manual files, run:

```bash
repogen setup-resources --config configs/resources.yaml --target-dir .
```

You should see:
- `lincs_gene_info` as `generated` or `ok`
- `metabrain_cortex_normalized` as `generated` or `ok`
- no missing resources

If any file is still reported missing:
- confirm exact filename and path
- rerun the command

## 7) Recommended config alignment after resource install

Ensure your `configs/reference.yaml` points to installed resources. At minimum:
- `genome_dir`, `bfile_prefix`, `gene_loc_file`
- BioMart dictionary paths
- `predixcan_model_dir` and `predixcan_covariance_dir`

Recommended additions in `reference` section:

```yaml
ncbi_gene_info: "resources/reference/gene_info.gz"
ncbi_gene_history: "resources/reference/gene_history.gz"
```

Also ensure GWAS input is set in `configs/config.yaml`:
- `study.gwas_input: "resources/gwas/<your_file>.gz"`

## 8) HPC-specific workflow

Use this order on HPC:

1. Clone repo on project storage.
2. Create/activate the conda env.
3. Run automatic setup on a login node.
4. Download browser-only/manual files either:
   - directly on login node if possible, or
   - on local machine, then transfer with `scp` or `rsync`.
5. Place files under the same relative `resources/...` paths in the HPC repo.
6. Rerun `setup-resources` to generate derived outputs.

Transfer examples from your local machine to HPC:

```bash
scp resources/drug_signatures/geneinfo_beta.txt user@hpc:/path/to/RepoGen/resources/drug_signatures/
rsync -avP resources/eqtl/metabrain/ user@hpc:/path/to/RepoGen/resources/eqtl/metabrain/
```

Important:
- run `scp` from the machine that has the source file
- do not run `scp` on HPC while pointing to `C:\...` Windows paths

## 9) Final quick checklist

Run:

```bash
repogen setup-resources --config configs/resources.yaml --target-dir .
repogen validate --config configs/config.yaml --reference configs/reference.yaml
```

When complete:
- `setup-resources` should report resources as `ok` (and no unresolved missing items)
- `validate` should report configuration as valid
