# Configuration

RepoGen is driven by two YAML files:

- `configs/config.yaml` - study metadata and analysis parameters
- `configs/reference.yaml` - filesystem paths to reference data

Keep them together. If you pass only `--config`, RepoGen auto-discovers a
`reference.yaml` sitting next to it.

Every option is validated against a Pydantic schema before anything runs, so a
typo or an out-of-range value fails immediately with a message naming the
field, rather than surfacing as a confusing error an hour into a run.

```bash
repogen validate --config configs/config.yaml
```

`validate` also checks that the external data your selected branches need is
actually on disk. Add `--strict` to make missing optional resources an error
too.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `study` | **required** | Study metadata (below) |
| `resource_dir` | `resources` | Root for downloaded reference data |
| `output_dir` | `results` | Root for all outputs |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `log_file` | `null` | Also write logs to this path |

Both `resource_dir` and `output_dir` may be absolute. On HPC, point them at
scratch storage while the code lives in your home directory.

## `study`

```yaml
study:
  name: PGC3_SCZ_wave3           # required; names the output subdirectory
  gwas_input: data/scz.tsv.gz    # required
  sample_size: 161405
  n_cases: 53386
  n_controls: 77258
  genome_build: GRCh37
  trait_type: binary             # binary | continuous
  population_prevalence: 0.01
  description: "PGC schizophrenia wave 3"
```

Only `name` and `gwas_input` are required. The rest improve the analysis where
supplied:

- `sample_size` is needed by MAGMA when the GWAS has no per-variant N column.
- `n_cases` / `n_controls` let colocalisation model a case-control trait
  correctly instead of assuming a quantitative one.
- `population_prevalence` is the **true lifetime population** prevalence, not
  the case fraction in your sample. It is used for Steiger filtering in
  Branch C. Leave it unset rather than guessing.
- `genome_build` is auto-detected when omitted, but stating it avoids an
  expensive misdetection.
- `trait_type` is `binary` or `continuous`. It decides how colocalisation
  models the outcome, so a mislabelled continuous trait will produce
  misleading posteriors.

Column names in the GWAS file are auto-detected across the common consortium
formats (PGC, UK Biobank, FinnGen, METAL).

## `gwas_prep`

Quality control applied once, to the shared standardised GWAS that every
branch reads.

| Key | Default | Meaning |
|---|---|---|
| `info_threshold` | `0.6` | Drop variants below this imputation INFO score |
| `maf_threshold` | `0.01` | Drop variants below this minor allele frequency |
| `liftover_to` | `null` | Target build for liftover; `null` means no liftover |
| `remove_mhc` | `false` | Drop chr6:25-35Mb here, for every branch |

`remove_mhc` is deliberately separate from `magma.exclude_mhc`. This one strips
the MHC from the shared input, so no branch ever sees it. The MAGMA setting
excludes it from Branch A's gene analysis only, which is usually what you want:
Branch C reports an MHC sensitivity analysis of its own, and stripping the
region upstream would make that impossible.

Branch B needs GRCh38 coordinates. If your GWAS is GRCh37, the workflow lifts
it over for Branch B specifically, leaving Branch A on the original build.

## Branch A

### `magma`

| Key | Default | Meaning |
|---|---|---|
| `binary_path` | `null` | Explicit MAGMA path; wins over auto-detection |
| `annotation_mode` | `proximity` | `proximity` or an H-MAGMA chromatin annotation |
| `gene_model` | `mean` | MAGMA gene-level model |
| `window_upstream_kb` | `35` | Gene window, upstream |
| `window_downstream_kb` | `10` | Gene window, downstream |
| `exclude_mhc` | `true` | Drop the extended MHC region |

The asymmetric 35/10 kb window is MAGMA's own recommended default; it reflects
that regulatory elements cluster upstream of transcription start sites.

Leave `exclude_mhc` on unless you have a specific reason. The MHC has such
extensive long-range LD that gene-level statistics there are not comparable to
the rest of the genome.

### `pathway`

Gene-set analysis, run between the gene step and drug enrichment.

| Key | Default | Meaning |
|---|---|---|
| `sources` | `null` | MSigDB collections to test; `null` uses everything found |
| `min_set_size` | `10` | Ignore gene sets smaller than this |
| `max_set_size` | `500` | Ignore gene sets larger than this |
| `fdr_method` | `fdr_bh` | Multiple-testing correction |
| `fdr_threshold` | `0.05` | Significance threshold |
| `driver_gene_p_threshold` | `0.05` | Gene-level p below which a gene is reported as driving its set |
| `extra_gmt_files` | `[]` | Additional GMT files to test alongside MSigDB |

The size bounds matter more than they look. Very small sets are unstable and
very large ones are so unspecific that significance says little, so both tails
are excluded by default rather than corrected for afterwards.

### `drug_enrichment`

| Key | Default | Meaning |
|---|---|---|
| `sources` | `[chembl]` | Also `pdsp`, `dgidb` |
| `chembl_scope` | `mechanism_only` | `mechanism_only` or include affinity data |
| `min_genes_per_drug` | `3` | Headline threshold: drugs with fewer targets are reported but not headlined |
| `atc_min_genes_per_drug` | `null` | Inherits `min_genes_per_drug`; set lower to widen the ATC pool |
| `min_pchembl` | `null` | Binding-affinity cutoff (pChEMBL) |
| `max_phase_filter` | `null` | Restrict by clinical phase |
| `phase_filter_scope` | `global` | `global` applies the phase filter to every source; `chembl_only` exempts PDSP and DGIdb |
| `confidence_filter` | `null` | Minimum interaction confidence |
| `pdsp_dedup_mode` | `off` | `exact_signature` collapses PDSP-only drugs with identical target sets |
| `parent_salt_unification` | `true` | Collapse child and salt ChEMBL IDs onto the parent compound |
| `enable_expression_enrichment` | `false` | Opt in to expression-perturbation evidence (CREEDS, DSigDB D3) |
| `expression_sources` | `[]` | Which expression sources to use when the above is on |
| `include_wilcoxon_auc` | `true` | Report a non-parametric concordance check alongside the main test |
| `fdr_method` | `fdr_bh` | Multiple-testing correction |
| `fdr_threshold` | `0.05` | Significance threshold |
| `permutation_test` | `false` | Empirical null by drug-label permutation |
| `n_permutations` | `10000` | Permutations when enabled |
| `permutation_seed` | `42` | Fixed for reproducibility |

`parent_salt_unification` is on by default because ChEMBL lists salt forms
under separate IDs. Without it, one drug can appear several times and dilute
its own enrichment signal.

`phase_filter_scope` exists because clinical phase is a ChEMBL field. Applying
a phase cutoff `global`ly silently discards every PDSP and DGIdb drug, since
they have no phase to compare. Use `chembl_only` to filter ChEMBL by phase
while keeping the other sources.

Turning on `enable_expression_enrichment` requires the DSigDB D3 resource,
which `setup-resources` will not download unless the flag is set. This is
deliberate default-deny: the resource is only fetched when you have opted in.

`atc_min_genes_per_drug` must not exceed `min_genes_per_drug`; the schema
rejects that combination. The intent is a *more* inclusive pool for
drug-class enrichment than for individual-drug headlines, since class-level
tests aggregate weak per-drug evidence.

Enabling `permutation_test` is substantially slower but gives a calibrated
null when the parametric assumptions are doubtful.

### `atc_enrichment`

Tests whole drug classes rather than individual drugs, so weak per-drug
evidence can aggregate into a class-level signal.

| Key | Default | Meaning |
|---|---|---|
| `atc_levels` | `[2, 3]` | ATC hierarchy levels to test, 1 to 4 |
| `min_drugs_per_class` | `5` | Skip classes with fewer drugs than this |
| `two_sided` | `false` | Default is one-sided upper-tail, i.e. enrichment only |
| `atc_universe_mode` | `annotated_only` | Which drugs form the denominator |
| `custom_classes` | `[]` | Curated drug sets tested as extra classes |
| `eigenvalue_threshold` | `0.1` | Eigenvalue clipping when regularising the correlation matrix |
| `fdr_method` | `fdr_bh` | Multiple-testing correction |
| `fdr_threshold` | `0.05` | Significance threshold |
| `permutation_test` | `false` | Drug-label permutation null |
| `n_permutations` | `10000` | Permutations when enabled |
| `permutation_seed` | `42` | Fixed for reproducibility |

Levels 2 and 3 are the default because level 1 is too coarse to be actionable
(14 anatomical groups) and level 4 too granular to accumulate enough drugs per
class. `two_sided` stays off because the hypothesis is directional: you are
asking whether a class is *enriched*, not whether it differs in either
direction.

`atc_universe_mode: annotated_only` restricts the denominator to drugs that
carry an ATC code, matching the DRUGSETS convention. Widening it changes what
significance means, so change it only deliberately.

`custom_classes` is additive: your curated sets are tested alongside the real
ATC classes, not instead of them.

## Branch B

### `spredixcan`

| Key | Default | Meaning |
|---|---|---|
| `model_type` | `mashr` | PredictDB model family |
| `tissue_preset` | `brain_13` | Named tissue set |
| `tissues` | preset | Explicit tissue list, overrides the preset |
| `extra_models` | `[]` | Additional PredictDB model files beyond the preset |
| `gwas_imputation` | `false` | Impute missing GWAS variants before association |
| `min_snps_used_fraction` | `0.1` | Drop genes with too few model SNPs matched |
| `exclude_palindromic` | `true` | Skip strand-ambiguous variants |

`exclude_palindromic` should stay on unless your GWAS and models are known to
share a strand convention; palindromic SNPs are a classic source of sign
errors that silently invert an effect direction.

### `drug_signatures`

How the many LINCS profiles per drug are collapsed into one signature.

| Key | Default | Meaning |
|---|---|---|
| `aggregation` | `consensus` | `consensus` (median across profiles), `best_dose`, or `per_condition` |
| `min_profiles` | `3` | Minimum profiles required to keep a drug |
| `preferred_dose` | `10 µM` | Target dose for `best_dose` |
| `preferred_time` | `24 h` | Target time point for `best_dose` |
| `cell_line_weighting` | `uniform` | `uniform`, `neural_priority`, or `neural_only` |
| `include_neural_tumor_cell_lines` | `true` | Include neural-lineage cancer lines in the derived default |
| `neural_cell_lines` | `null` | Explicit cell-line tokens; overrides the switch above |
| `neural_weight` | `3.0` | Upweight applied to neural profiles under `neural_priority` |
| `min_neural_profiles` | `null` | Minimum neural profiles under `neural_only`; inherits `min_profiles` |

`cell_line_weighting: uniform` reproduces the pre-weighting behaviour exactly,
byte for byte, and a regression test locks that. Change it only if you
specifically want brain-relevant cell lines to dominate the consensus.

Setting `neural_cell_lines` explicitly *bypasses*
`include_neural_tumor_cell_lines` entirely rather than combining with it. The
defaults were validated against the shipped LINCS corpus, and tokens with zero
observed profiles were excluded; `scripts/lincs_cell_line_census.py` re-checks
this against any GCTX file and exits non-zero if a shipped default is absent.

### `negative_correlation`

Scores how strongly each drug signature reverses the disease signature.

| Key | Default | Meaning |
|---|---|---|
| `correlation_method` | `spearman` | Only `spearman` is supported in v1.0 |
| `min_overlapping_genes` | `50` | Minimum shared genes before a correlation is computed |
| `gene_set_mode` | `landmark` | Which gene space to correlate over |
| `match_confidence_threshold` | `pubchem_cid` | Minimum drug-match confidence: `inchikey` > `pubchem_cid` > `name` |
| `min_profiles_aggregated` | `3` | Minimum LINCS profiles behind an included drug |
| `aggregation_mode` | `consensus` | Which `drug_signatures` mode to consume |
| `exclude_mhc` | `false` | Drop MHC genes from the correlation |
| `xsum_top_n` | `200` | Top N genes for the XSum statistic |
| `xsum_permutations` | `0` | Permutations for an XSum p-value; 0 disables |
| `xsum_seed` | `42` | Seed; the default preserves earlier behaviour byte for byte |
| `fdr_threshold` | `0.05` | Significance threshold |

`match_confidence_threshold` is the one to think about. InChIKey matching is
near-exact; name matching is fuzzy and will pair compounds that merely have
similar names. The default sits in the middle: PubChem CID or better.

FDR is applied per tissue, not once globally, because tissues differ in power
and a single pooled correction would let well-powered tissues dominate. A
calibration sidecar reports genomic-control lambda so you can see whether the
per-tissue nulls are behaving.

## Branch C

### `mr`

| Key | Default | Meaning |
|---|---|---|
| `eqtl_sources` | - | `eqtlgen`, `metabrain`, or both; each has `source`, `path`, and optionally `allele_frequency_path`, `required`, `min_result_fraction` |
| `eqtl_sources[].allele_frequency_path` | `null` | eQTLGen only: its allele-frequency file (setup-resources `eqtlgen_allele_frequency`); without it the 1000 Genomes MAF is used and palindromic SNPs are dropped |
| `cis_window_kb` | `1000` | cis region around the gene position the eQTL source reports |
| `instrument_pval` | `5e-08` | Instrument selection threshold |
| `clump_r2` | `0.001` | LD clumping threshold |
| `f_stat_threshold` | `10.0` | Weak-instrument filter |
| `coloc_enabled` | `true` | Run colocalisation |
| `coloc_variance_mode` | `reported_se` | Outcome variance source |
| `coloc_pp_h4_threshold` | `0.8` | Shared-causal-variant posterior |
| `min_coloc_snps` | `50` | Minimum overlap to attempt coloc |
| `require_coloc` | `true` | Causal calls must also colocalise |
| `require_steiger` | `false` | Enforce direction-of-effect filter |
| `coloc_prior_p1` | `1e-4` | Prior that a variant affects expression |
| `coloc_prior_p2` | `1e-4` | Prior that a variant affects the trait |
| `coloc_prior_p12` | `1e-5` | Prior that a variant affects both |
| `drug_match` | see below | Filters for joining MR targets to drugs |
| `mhc_sensitivity` | see below | Repeat the analysis with MHC genes dropped |
| `n_workers` | `1` | Parallel workers for the gene loop |
| `rule_threads` | `null` | CPUs requested for the MR rule; defaults to the rule's own budget |

Instruments are chosen among SNPs the GWAS can use. Every candidate is matched
to the GWAS first, by rsID, and by position only when the eQTL source is on
the GWAS's genome build, with matching alleles. A palindromic SNP (A/T or
C/G) is kept only when the eQTL and GWAS frequencies of its effect allele
(MetaBrain's `eaf`, eQTLGen's allele-frequency file, the GWAS control
frequency `FCON`) are both at least 0.08 from 0.5 and on the same side of
it. Branch C therefore reads its own GWAS preparation,
`prepare_data/gwas_mr.parquet`, which skips reference-panel harmonisation,
since that step removes every palindromic SNP.

LD clumping ranks the usable SNPs by eQTL |z| (eQTLGen floors its P values,
so ranking by P left ties to file order). PLINK clumps only SNPs in the LD
reference panel, so when a gene has several usable SNPs its instruments are
the clumped panel SNPs and the lead instrument is the strongest of them; a
gene with a single usable SNP keeps it whether or not the panel has it.

The `F > 10` convention guards against weak-instrument bias. `require_coloc`
defaults to on because an MR estimate without colocalisation cannot distinguish
a genuinely shared causal variant from two distinct variants in LD, which is
the most common way cis-MR produces false positives.

Branch C is the slowest stage. On CREATE, a full PGC3 schizophrenia run over
17,189 genes takes about **2.1 hours** with 4 workers (about 330 genes/min),
peaking at 9.7 GB in the Python process. The first run also writes a
per-chromosome copy of the LD reference panel beside it, which takes a few
minutes once. Raise `n_workers` (with `rule_threads` to match) if your
scheduler will give you the cores.

`mhc_sensitivity` reruns the whole analysis with MHC genes dropped and writes
it to `mr/sensitivity/mhc_excluded/`. It is worth leaving on: in the run above
it showed 215 significant genes overall versus 179 outside the MHC, while the
high-confidence count was 25 either way, which is exactly the reassurance you
want that the headline result is not an artefact of MHC long-range LD.

Steiger filtering is skipped for any gene when `study.population_prevalence`
is unset, and the metadata sidecar records how many were skipped for that
reason. In the run above that was 23,761. If you intend to rely on Steiger,
set the prevalence.

## `open_targets`

Optional annotation of results with known disease associations.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Annotate results via Open Targets |
| `disease_efo_id` | `null` | EFO term for your disease, e.g. `EFO_0003761` |
| `use_api` | `true` | Query the GraphQL API rather than a bulk download |
| `cache_dir` | `null` | Where to cache API responses |

Set `disease_efo_id` or this does nothing useful. On a compute node with no
outbound network, set `enabled: false` or pre-populate `cache_dir` from a
login-node run, otherwise the rule will stall on API calls.

## `output`

Which artefacts to produce.

| Key | Default | Meaning |
|---|---|---|
| `csv` | `true` | Write CSV exports |
| `excel` | `true` | Write XLSX exports |
| `html` | `true` | Build the HTML report |
| `manhattan` | `true` | Manhattan plot |
| `qq` | `true` | QQ plot |
| `enrichment` | `true` | Enrichment figures |
| `plot_style` | see below | Figure styling |

Excel has a hard 32,767-character cell limit. Exports containing long
concatenated gene lists are truncated at that boundary and the truncation is
logged with the offending column, so check the log if a spreadsheet cell looks
cut off. The CSV and JSON exports are not truncated; prefer them for anything
downstream.

## Running on HPC

Points that only show up on a cluster:

- **One driver per working directory.** Snakemake locks the working directory.
  A second `repogen run` against the same checkout fails with a
  `LockException` rather than corrupting anything. To run two studies at once,
  give each its own checkout.
- **The merged config is written under `output_dir/.repogen/`, not `/tmp`.**
  Every submitted job re-reads it from a compute node, and `/tmp` is
  node-local, so a temp-directory config makes cluster jobs fail with
  `FileNotFoundError`.
- **Keep the scheduler's retry and Snakemake's retry from fighting.** The
  supplied profile passes `--no-requeue`. Without it, a node failure is
  retried twice over: SLURM requeues the original job while Snakemake, having
  already written that job off, submits a replacement. Both then run the same
  rule and write the same outputs at once, and the requeued one is invisible
  to Snakemake even if it succeeds.
- **Point `resource_dir` and `output_dir` at scratch**, with the code in your
  home directory. Scratch is usually much larger and faster, and often not
  backed up, which is the right place for regenerable intermediates.
- **Recovering from an interrupted run:** `repogen run ... -- --unlock`, then
  rerun. Use the passthrough form rather than calling `snakemake --unlock`
  directly, since the config is assembled by `repogen`.

## `reference.yaml`

Paths to reference data, normally written for you by
`repogen setup-resources`. Edit it directly if your data sits somewhere the
resource manager did not put it. `repogen validate` reports any path here that
does not resolve.

## Precedence

For MAGMA specifically, the binary is resolved as:

1. `magma.binary_path` in your config
2. `<resource_dir>/bin/magma`
3. whatever is on `PATH`

An explicitly configured path therefore always wins, and a `setup-resources`
installation is found without modifying your environment.
