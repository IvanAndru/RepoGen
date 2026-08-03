"""Tests for repogen.config.schema and repogen.config.loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from repogen.config.loader import load_config
from repogen.config.schema import (
    DrugEnrichmentConfig,
    GWASPrepConfig,
    MagmaConfig,
    NegativeCorrelationConfig,
    PipelineConfig,
    StudyConfig,
)
from repogen.utils.constants import BRAIN_TISSUES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> dict:
    """Return a minimal valid config dict."""
    return {
        "study": {
            "name": "test_study",
            "gwas_input": "data/test.gwas.gz",
        },
    }


def _full_config() -> dict:
    """Return a fully populated config dict."""
    base = _minimal_config()
    base["study"]["sample_size"] = 50000
    base["study"]["genome_build"] = "GRCh37"
    base["study"]["trait_type"] = "case_control"
    base["gwas_prep"] = {"info_threshold": 0.8, "maf_threshold": 0.05}
    base["magma"] = {"window_upstream_kb": 50, "window_downstream_kb": 15}
    base["drug_enrichment"] = {"sources": ["chembl", "pdsp"], "n_permutations": 5000}
    base["negative_correlation"] = {"tissues": ["Brain_Cortex"]}
    base["log_level"] = "DEBUG"
    return base


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data, default_flow_style=False))


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for Pydantic schema models."""

    def test_minimal_config_valid(self) -> None:
        config = PipelineConfig(**_minimal_config())
        assert config.study.name == "test_study"

    def test_full_config_valid(self) -> None:
        config = PipelineConfig(**_full_config())
        assert config.study.sample_size == 50000
        assert config.gwas_prep.info_threshold == 0.8

    def test_resource_dir_defaults_to_resources(self) -> None:
        """Omitting resource_dir keeps the working-directory-relative default."""
        config = PipelineConfig(**_minimal_config())
        assert config.resource_dir == Path("resources")

    def test_resource_dir_accepts_absolute_path(self) -> None:
        """Shared cluster storage is addressed by an absolute resource root."""
        cfg = _minimal_config()
        cfg["resource_dir"] = "/scratch/prj/example/resources"
        config = PipelineConfig(**cfg)
        assert config.resource_dir == Path("/scratch/prj/example/resources")

    def test_resource_dir_survives_serialisation(self) -> None:
        """The CLI hands Snakemake a serialised config, so the key must survive.

        Pydantic drops unknown keys, so a resource_dir absent from the schema
        would silently revert every workflow path to the default root.
        """
        cfg = _minimal_config()
        cfg["resource_dir"] = "/scratch/prj/example/resources"
        dumped = PipelineConfig(**cfg).model_dump(mode="json")
        # Compared as Path so the assertion holds on Windows, where the
        # serialised form uses backslashes.
        assert Path(dumped["resource_dir"]) == Path("/scratch/prj/example/resources")

    def test_study_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            StudyConfig(gwas_input="data/test.gz")

    def test_study_missing_gwas_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            StudyConfig(name="test")

    def test_study_invalid_genome_build_raises(self) -> None:
        with pytest.raises(ValidationError):
            StudyConfig(name="test", gwas_input="x.gz", genome_build="hg19")

    def test_study_valid_genome_builds(self) -> None:
        s37 = StudyConfig(name="t", gwas_input="x.gz", genome_build="GRCh37")
        s38 = StudyConfig(name="t", gwas_input="x.gz", genome_build="GRCh38")
        assert s37.genome_build == "GRCh37"
        assert s38.genome_build == "GRCh38"

    def test_study_none_genome_build_ok(self) -> None:
        s = StudyConfig(name="t", gwas_input="x.gz", genome_build=None)
        assert s.genome_build is None

    def test_gwas_prep_info_threshold_range(self) -> None:
        GWASPrepConfig(info_threshold=0.0)
        GWASPrepConfig(info_threshold=1.0)
        with pytest.raises(ValidationError):
            GWASPrepConfig(info_threshold=-0.1)
        with pytest.raises(ValidationError):
            GWASPrepConfig(info_threshold=1.5)

    def test_gwas_prep_maf_threshold_range(self) -> None:
        GWASPrepConfig(maf_threshold=0.0)
        GWASPrepConfig(maf_threshold=0.5)
        with pytest.raises(ValidationError):
            GWASPrepConfig(maf_threshold=-0.01)
        with pytest.raises(ValidationError):
            GWASPrepConfig(maf_threshold=0.6)

    def test_magma_window_non_negative(self) -> None:
        MagmaConfig(window_upstream_kb=0, window_downstream_kb=0)
        MagmaConfig(window_upstream_kb=100, window_downstream_kb=50)
        with pytest.raises(ValidationError):
            MagmaConfig(window_upstream_kb=-1)
        with pytest.raises(ValidationError):
            MagmaConfig(window_downstream_kb=-1)

    def test_drug_enrichment_n_permutations_min(self) -> None:
        DrugEnrichmentConfig(n_permutations=100)
        with pytest.raises(ValidationError):
            DrugEnrichmentConfig(n_permutations=50)

    # --- DrugEnrichmentConfig.sources validation ---

    def test_drug_sources_default_is_chembl(self) -> None:
        de = DrugEnrichmentConfig()
        assert de.sources == ["chembl"]

    def test_drug_sources_accepts_all_valid(self) -> None:
        de = DrugEnrichmentConfig(sources=["chembl", "pdsp", "dgidb"])
        assert de.sources == ["chembl", "pdsp", "dgidb"]

    def test_drug_sources_normalizes_case(self) -> None:
        de = DrugEnrichmentConfig(sources=["ChEMBL", "PDSP", "DGIdb"])
        assert de.sources == ["chembl", "pdsp", "dgidb"]

    def test_drug_sources_deduplicates(self) -> None:
        de = DrugEnrichmentConfig(sources=["chembl", "CHEMBL", "pdsp"])
        assert de.sources == ["chembl", "pdsp"]

    def test_drug_sources_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError, match="Unknown drug source"):
            DrugEnrichmentConfig(sources=["chembl", "drugbank"])

    def test_drug_sources_rejects_empty(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            DrugEnrichmentConfig(sources=[])

    # --- DrugEnrichmentConfig.chembl_scope validation ---

    def test_chembl_scope_default(self) -> None:
        de = DrugEnrichmentConfig()
        assert de.chembl_scope == "mechanism_only"

    def test_chembl_scope_accepts_valid(self) -> None:
        de = DrugEnrichmentConfig(chembl_scope="mechanism_or_affinity")
        assert de.chembl_scope == "mechanism_or_affinity"

    def test_chembl_scope_normalizes_case(self) -> None:
        de = DrugEnrichmentConfig(chembl_scope="Mechanism_Only")
        assert de.chembl_scope == "mechanism_only"

    def test_chembl_scope_rejects_invalid(self) -> None:
        with pytest.raises(ValidationError, match="chembl_scope"):
            DrugEnrichmentConfig(chembl_scope="all")

    def test_negative_correlation_default_tissues(self) -> None:
        nc = NegativeCorrelationConfig()
        assert len(nc.tissues) == len(BRAIN_TISSUES)
        assert nc.tissues == list(BRAIN_TISSUES)

    def test_negative_correlation_custom_tissues(self) -> None:
        nc = NegativeCorrelationConfig(tissues=["Brain_Cortex", "Brain_Hippocampus"])
        assert nc.tissues == ["Brain_Cortex", "Brain_Hippocampus"]

    def test_invalid_log_level_raises(self) -> None:
        cfg = _minimal_config()
        cfg["log_level"] = "VERBOSE"
        with pytest.raises(ValidationError):
            PipelineConfig(**cfg)

    def test_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = _minimal_config()
            cfg["log_level"] = level
            config = PipelineConfig(**cfg)
            assert config.log_level == level

    # --- MagmaConfig Phase 3 tests (24-28) ---

    def test_magma_config_defaults(self) -> None:
        mc = MagmaConfig()
        assert mc.annotation_mode == "proximity"
        assert mc.gene_model == "mean"
        assert mc.window_upstream_kb == 35
        assert mc.window_downstream_kb == 10
        assert mc.exclude_mhc is True
        assert mc.binary_path is None
        assert mc.custom_annot_file is None
        assert mc.biotype_filter is None
        assert mc.memory_efficient is False

    def test_magma_config_invalid_annotation_mode(self) -> None:
        with pytest.raises(ValidationError, match="annotation_mode"):
            MagmaConfig(annotation_mode="invalid_mode")

    def test_magma_config_invalid_gene_model(self) -> None:
        with pytest.raises(ValidationError, match="gene_model"):
            MagmaConfig(gene_model="top_snp")

    def test_magma_config_custom_requires_path(self) -> None:
        with pytest.raises(ValidationError, match="custom_annot_file"):
            MagmaConfig(annotation_mode="custom")

    def test_magma_config_old_fields_removed(self) -> None:
        mc = MagmaConfig()
        assert not hasattr(mc, "chromosome_by_chromosome")
        assert not hasattr(mc, "gene_window_kb")


class TestExpressionEnrichmentConfig:
    """enable_expression_enrichment + expression_sources."""

    # --- defaults ---

    def test_expression_enrichment_off_by_default(self) -> None:
        de = DrugEnrichmentConfig()
        assert de.enable_expression_enrichment is False
        assert de.expression_sources == []

    # --- expression_sources field validator ---

    def test_expression_sources_accepts_valid(self) -> None:
        de = DrugEnrichmentConfig(
            enable_expression_enrichment=True,
            expression_sources=["creeds", "dsigdb"],
        )
        assert de.expression_sources == ["creeds", "dsigdb"]

    def test_expression_sources_normalises_case(self) -> None:
        de = DrugEnrichmentConfig(
            enable_expression_enrichment=True,
            expression_sources=["CREEDS", "DSigDB"],
        )
        assert de.expression_sources == ["creeds", "dsigdb"]

    def test_expression_sources_deduplicates(self) -> None:
        de = DrugEnrichmentConfig(
            enable_expression_enrichment=True,
            expression_sources=["creeds", "CREEDS", "dsigdb"],
        )
        assert de.expression_sources == ["creeds", "dsigdb"]

    def test_expression_sources_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError, match="Unknown expression source"):
            DrugEnrichmentConfig(
                enable_expression_enrichment=True,
                expression_sources=["creeds", "lincs"],
            )

    # --- model-level invariants ---

    def test_enable_expression_requires_non_empty_sources(self) -> None:
        with pytest.raises(ValidationError, match="non-empty expression_sources"):
            DrugEnrichmentConfig(
                enable_expression_enrichment=True,
                expression_sources=[],
            )

    def test_disjoint_set_rejects_overlap_via_construct(self) -> None:
        # Use the per-field validator path: target-only "chembl" is
        # invalid as expression source, and vice versa.  This covers
        # the strict-domain enforcement which makes overlap impossible
        # in normal Pydantic construction.
        with pytest.raises(ValidationError, match="Unknown expression source"):
            DrugEnrichmentConfig(
                enable_expression_enrichment=True,
                expression_sources=["chembl"],   # target-family token
            )
        with pytest.raises(ValidationError, match="Unknown drug source"):
            DrugEnrichmentConfig(sources=["creeds"])  # expression-family token

    def test_partial_subset_creeds_only_allowed(self) -> None:
        de = DrugEnrichmentConfig(
            enable_expression_enrichment=True,
            expression_sources=["creeds"],
        )
        assert de.expression_sources == ["creeds"]

    def test_partial_subset_dsigdb_only_allowed(self) -> None:
        de = DrugEnrichmentConfig(
            enable_expression_enrichment=True,
            expression_sources=["dsigdb"],
        )
        assert de.expression_sources == ["dsigdb"]

    # --- backward compat ---

    def test_legacy_construction_unchanged(self) -> None:
        # Legacy code/tests instantiate DrugEnrichmentConfig without
        # touching the new fields.  Behaviour must be byte-identical.
        de = DrugEnrichmentConfig(min_genes_per_drug=1)
        assert de.enable_expression_enrichment is False
        assert de.expression_sources == []
        assert de.min_genes_per_drug == 1


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the YAML config loader."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        _write_yaml(f, _minimal_config())
        config = load_config(f)
        assert config.study.name == "test_study"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_bare_list_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(f)

    def test_env_variable_expansion(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("TEST_REPOGEN_DIR", "/expanded/path")
        cfg = _minimal_config()
        cfg["study"]["gwas_input"] = "$TEST_REPOGEN_DIR/gwas.gz"
        f = tmp_path / "config.yaml"
        _write_yaml(f, cfg)
        config = load_config(f)
        assert config.study.gwas_input == Path("/expanded/path/gwas.gz")

    def test_reference_yaml_merge(self, tmp_path: Path) -> None:
        main_cfg = _minimal_config()
        ref_cfg = {
            "reference": {
                "population": "EAS",
                "genome_dir": "/ref/data",
            },
        }
        main_f = tmp_path / "config.yaml"
        ref_f = tmp_path / "reference.yaml"
        _write_yaml(main_f, main_cfg)
        _write_yaml(ref_f, ref_cfg)
        config = load_config(main_f, ref_f)
        assert config.reference.population == "EAS"

    def test_bad_config_values_raises(self, tmp_path: Path) -> None:
        cfg = _minimal_config()
        cfg["gwas_prep"] = {"info_threshold": 99.9}
        f = tmp_path / "config.yaml"
        _write_yaml(f, cfg)
        with pytest.raises(ValidationError):
            load_config(f)

    def test_empty_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ValidationError):
            load_config(f)

    def test_auto_discover_sibling_reference(self, tmp_path: Path) -> None:
        """When reference_path is None and a sibling reference.yaml exists,
        it should be merged automatically."""
        main_cfg = _minimal_config()
        ref_cfg = {
            "reference": {
                "population": "SAS",
                "gene_loc_file": "/data/gene.loc",
            },
        }
        _write_yaml(tmp_path / "config.yaml", main_cfg)
        _write_yaml(tmp_path / "reference.yaml", ref_cfg)
        config = load_config(tmp_path / "config.yaml")
        assert config.reference.population == "SAS"
        assert config.reference.gene_loc_file == Path("/data/gene.loc")

    def test_no_auto_discover_when_sibling_absent(self, tmp_path: Path) -> None:
        """When reference_path is None and no sibling reference.yaml exists,
        defaults should be used (no error)."""
        _write_yaml(tmp_path / "config.yaml", _minimal_config())
        config = load_config(tmp_path / "config.yaml")
        assert config.reference.gene_loc_file is None
        assert config.reference.population == "EUR"

    def test_explicit_reference_overrides_auto(self, tmp_path: Path) -> None:
        """An explicit reference_path should take priority over a sibling
        reference.yaml that also exists."""
        main_cfg = _minimal_config()
        sibling_ref = {"reference": {"population": "AFR"}}
        explicit_ref = {"reference": {"population": "EAS"}}
        _write_yaml(tmp_path / "config.yaml", main_cfg)
        _write_yaml(tmp_path / "reference.yaml", sibling_ref)
        explicit_f = tmp_path / "other_ref.yaml"
        _write_yaml(explicit_f, explicit_ref)
        config = load_config(tmp_path / "config.yaml", explicit_f)
        assert config.reference.population == "EAS"


class TestClusterProfileContract:
    """Guard the HPC submission contract.

    Snakemake 8 removed --cluster-config in favour of executor plugins, so
    cluster settings live in a profile and per-rule budgets live on the rules
    themselves. Rules previously carried `partition`/`rule_name` entries that
    no version of Snakemake reads: jobs submitted that way silently inherit
    the scheduler default (on CREATE, 1 core / 1 GB / 24 h) and the
    memory-heavy rules are killed.
    """

    _PROFILE = Path("profiles/create/config.yaml")
    _RULES = sorted(Path("workflows/rules").glob("*.smk"))

    def test_profile_exists_and_selects_slurm(self) -> None:
        cfg = yaml.safe_load(self._PROFILE.read_text(encoding="utf-8"))
        assert cfg["executor"] == "slurm"
        assert cfg["default-resources"]["slurm_partition"]
        assert cfg["jobs"] >= 1

    def test_no_rule_uses_the_dead_resource_keys(self) -> None:
        offenders = []
        for path in self._RULES:
            text = path.read_text(encoding="utf-8")
            if 'partition="general"' in text or "rule_name=" in text:
                offenders.append(path.name)
        assert not offenders, f"inert cluster keys resurfaced in: {offenders}"

    def test_every_resources_block_declares_runtime_and_memory(self) -> None:
        """A block without these submits with scheduler defaults and dies."""
        missing = []
        for path in self._RULES:
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "resources:":
                    continue
                block = []
                for nxt in lines[i + 1:]:
                    if nxt.strip() and not nxt.startswith((" " * 8, "\t\t")):
                        break
                    block.append(nxt)
                joined = "\n".join(block)
                if "runtime=" not in joined or "mem_mb=" not in joined:
                    missing.append(f"{path.name}:{i + 1}")
        assert not missing, f"resources blocks without runtime/mem_mb: {missing}"

    def test_retired_cluster_yaml_is_gone(self) -> None:
        """configs/cluster.yaml targeted an interface Snakemake 8 removed."""
        assert not Path("configs/cluster.yaml").exists()


@pytest.mark.skipif(
    not Path("envs/Dockerfile").exists(),
    reason="deployment files are not shipped inside the image; repo checkout only",
)
class TestContainerImageContract:
    """Guard the deployment artefacts that travel to other people's clusters.

    The image is the supported way for collaborators to run RepoGen, so the
    ways it can silently become wrong - an unpinned base, a build that ignores
    the lock file, a redistributed MAGMA binary, or a build context that drags
    in ~186 GB of data - are asserted here rather than discovered in CI.
    """

    _DOCKERFILE = Path("envs/Dockerfile")
    _IGNORE = Path(".dockerignore")
    _CI = Path(".github/workflows/container.yml")

    def test_base_image_is_pinned(self) -> None:
        text = self._DOCKERFILE.read_text(encoding="utf-8")
        from_lines = [ln for ln in text.splitlines() if ln.startswith("FROM ")]
        assert from_lines, "Dockerfile has no FROM instruction"
        for line in from_lines:
            assert ":latest" not in line, f"unpinned base image: {line}"
            assert ":" in line.split()[1], f"base image lacks a tag: {line}"

    def test_environment_built_from_lock_file(self) -> None:
        """Building from envs/repogen.yaml would re-solve and drift."""
        text = self._DOCKERFILE.read_text(encoding="utf-8")
        assert "repogen.linux-64.lock" in text
        assert "repogen-pip.linux-64.txt" in text

    def test_magma_is_not_installed_in_the_image(self) -> None:
        """The MAGMA licence forbids redistributing its binaries."""
        text = self._DOCKERFILE.read_text(encoding="utf-8")
        install_lines = [
            ln for ln in text.splitlines()
            if ln.strip().startswith(("RUN", "COPY", "ADD"))
        ]
        for line in install_lines:
            assert "magma" not in line.lower(), (
                f"image build step appears to fetch MAGMA: {line}"
            )

    def test_dockerignore_excludes_data_directories(self) -> None:
        text = self._IGNORE.read_text(encoding="utf-8")
        for pattern in ["resources/", "results/", ".git/"]:
            assert pattern in text, f"{pattern} missing from .dockerignore"

    def test_ci_verifies_image_before_publishing(self) -> None:
        text = self._CI.read_text(encoding="utf-8")
        # The verification step must run before the push step.
        assert text.index("Verify the image actually works") < text.index("Push image")
        assert "repogen info" in text
        assert "pytest" in text
