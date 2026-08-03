"""Tests for repogen.data.resources - derived resource generation."""

from __future__ import annotations

import inspect
import os
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from repogen.data import resources as resources_mod
from repogen.data.resources import (
    _POSTPROCESSORS,
    _resolve_dotted_path,
    generate_lincs_gene_info,
    setup_resources,
)


class TestGenerateLincsGeneInfo:
    def test_primary_contract_pr_is_lm_pr_is_bing(self, tmp_path: Path):
        """Real geneinfo_beta.txt format: pr_is_lm/pr_is_bing columns with '1'/'0' values."""
        source = tmp_path / "geneinfo_beta.txt"
        source.write_text(
            "pr_gene_id\tpr_gene_symbol\tpr_gene_title\tpr_is_lm\tpr_is_bing\n"
            "1000\tABC\tAlpha\t1\t0\n"
            "2000\tDEF\tBeta\t0\t1\n"
            "3000\tGHI\tGamma\t0\t0\n"
        )
        out = tmp_path / "lincs_gene_info.tsv"
        generate_lincs_gene_info(source, out)

        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["entrez_id", "gene_symbol", "is_landmark", "is_bing"]
        assert len(df) == 3
        assert df.loc[0, "is_landmark"] == True
        assert df.loc[0, "is_bing"] == False
        assert df.loc[1, "is_landmark"] == False
        assert df.loc[1, "is_bing"] == True
        assert df.loc[2, "is_landmark"] == False
        assert df.loc[2, "is_bing"] == False

    def test_fallback_contract_pr_gene_space(self, tmp_path: Path):
        """Alternative file layout: pr_gene_space column with string labels."""
        source = tmp_path / "geneinfo_beta.txt"
        source.write_text(
            "pr_gene_id\tpr_gene_symbol\tpr_gene_space\n"
            "1000\tABC\tlandmark\n"
            "2000\tDEF\tbest inferred\n"
            "3000\tGHI\tinferred\n"
        )
        out = tmp_path / "lincs_gene_info.tsv"
        generate_lincs_gene_info(source, out)

        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["entrez_id", "gene_symbol", "is_landmark", "is_bing"]
        assert len(df) == 3
        assert df.loc[0, "is_landmark"] == True
        assert df.loc[0, "is_bing"] == False
        assert df.loc[1, "is_landmark"] == False
        assert df.loc[1, "is_bing"] == True
        assert df.loc[2, "is_landmark"] == False
        assert df.loc[2, "is_bing"] == False

    def test_no_classification_columns_raises(self, tmp_path: Path):
        """File has pr_gene_id/pr_gene_symbol but no classification columns."""
        source = tmp_path / "geneinfo_beta.txt"
        source.write_text(
            "pr_gene_id\tpr_gene_symbol\tpr_gene_title\n"
            "1000\tABC\tAlpha\n"
        )
        out = tmp_path / "out.tsv"

        with pytest.raises(ValueError, match="Cannot determine gene-set classification"):
            generate_lincs_gene_info(source, out)

    def test_missing_required_columns_raises(self, tmp_path: Path):
        source = tmp_path / "bad.txt"
        source.write_text("col_a\tcol_b\n1\t2\n")
        out = tmp_path / "out.tsv"

        with pytest.raises(ValueError, match="Expected gene ID columns"):
            generate_lincs_gene_info(source, out)


# ---------------------------------------------------------------------------
# Resource gating & DSigDB postprocessor
# ---------------------------------------------------------------------------


class TestResolveDottedPath:
    """Helper used by the gating contract - verify resolution semantics."""

    def test_resolves_simple_path(self) -> None:
        assert _resolve_dotted_path({"a": {"b": True}}, "a.b") is True

    def test_resolves_nested_path(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        assert _resolve_dotted_path(d, "a.b.c") == 42

    def test_returns_none_when_key_missing(self) -> None:
        assert _resolve_dotted_path({"a": {}}, "a.b") is None

    def test_returns_none_when_intermediate_missing(self) -> None:
        assert _resolve_dotted_path({}, "a.b.c") is None

    def test_returns_none_when_intermediate_not_dict(self) -> None:
        # Walking through a non-dict node (e.g. a list) returns None
        # rather than raising; gating treats this as a closed gate.
        assert _resolve_dotted_path({"a": ["x"]}, "a.b") is None


def _write_resources_yaml(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body))


def _write_pipeline_yaml(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body))


class TestResourceGating:
    """``enabled_by:`` gating contract.

    Critical-criteria checks:
      * Existing entries (no ``enabled_by:``) are unaffected, including
        ``optional: true`` entries (e.g. H-MAGMA continues to auto-
        download / verify).
      * Gated entries default-deny when ``pipeline_config`` is None.
      * Gated entries that resolve True via ``pipeline_config`` proceed
        through the normal download path.
      * Gated entries that resolve False via ``pipeline_config`` skip
        with status ``"gated_skipped"``.
    """

    def test_signature_legacy_three_arg_call_unchanged(self) -> None:
        # ``pipeline_config`` must be a kwarg with default None so
        # legacy callers keep working byte-identically.
        sig = inspect.signature(setup_resources)
        assert sig.parameters["pipeline_config"].default is None
        # The first three positional args must remain the same.
        positional = list(sig.parameters)
        assert positional[:3] == [
            "resources_config", "target_dir", "verify_only",
        ]

    def test_optional_without_enabled_by_still_downloads(
        self, tmp_path: Path,
    ) -> None:
        """H-MAGMA-style entry: ``optional: true`` without ``enabled_by:``.

        Documents the contract: ``optional:`` is a documentation-only
        marker today, and remains so with expression enrichment on.  The entry must
        proceed through the normal path (here: present-on-disk -> ok).
        """
        target = tmp_path / "fake_resource.txt"
        target.write_text("dummy")
        rcfg = tmp_path / "configs" / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "fake_optional": {
                    "url": "http://example.invalid/fake.txt",
                    "local_path": str(target.relative_to(tmp_path)),
                    "optional": True,
                    "description": "fake optional",
                },
            },
        })
        status = setup_resources(rcfg, target_dir=tmp_path)
        assert status["fake_optional"] == "ok"
        assert "gated_skipped" not in status.values()

    def test_gated_entry_default_denies_without_pipeline_config(
        self, tmp_path: Path,
    ) -> None:
        rcfg = tmp_path / "configs" / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "gated_thing": {
                    "url": "http://example.invalid/x.txt",
                    "local_path": "data/x.txt",
                    "optional": True,
                    "enabled_by": "drug_enrichment.enable_expression_enrichment",
                    "description": "should be skipped by default-deny",
                },
            },
        })
        status = setup_resources(rcfg, target_dir=tmp_path)
        assert status["gated_thing"] == "gated_skipped"

    def test_gated_entry_skips_when_flag_false(
        self, tmp_path: Path,
    ) -> None:
        rcfg = tmp_path / "configs" / "resources.yaml"
        pcfg = tmp_path / "configs" / "config.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "gated_thing": {
                    "url": "http://example.invalid/x.txt",
                    "local_path": "data/x.txt",
                    "optional": True,
                    "enabled_by": "drug_enrichment.enable_expression_enrichment",
                    "description": "gated",
                },
            },
        })
        _write_pipeline_yaml(pcfg, {
            "drug_enrichment": {"enable_expression_enrichment": False},
        })
        status = setup_resources(
            rcfg, target_dir=tmp_path, pipeline_config=pcfg,
        )
        assert status["gated_thing"] == "gated_skipped"

    def test_gated_entry_passes_through_when_flag_true(
        self, tmp_path: Path,
    ) -> None:
        # Pre-create the local file so the resource resolves to "ok"
        # and we don't issue a real HTTP download from the test.
        target = tmp_path / "data" / "x.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("dummy")

        rcfg = tmp_path / "configs" / "resources.yaml"
        pcfg = tmp_path / "configs" / "config.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "gated_thing": {
                    "url": "http://example.invalid/x.txt",
                    "local_path": "data/x.txt",
                    "optional": True,
                    "enabled_by": "drug_enrichment.enable_expression_enrichment",
                    "description": "gated",
                },
            },
        })
        _write_pipeline_yaml(pcfg, {
            "drug_enrichment": {"enable_expression_enrichment": True},
        })
        status = setup_resources(
            rcfg, target_dir=tmp_path, pipeline_config=pcfg,
        )
        # Gate is open -> falls through to existence check -> "ok".
        assert status["gated_thing"] == "ok"

    def test_gated_entry_treats_missing_flag_as_closed(
        self, tmp_path: Path,
    ) -> None:
        rcfg = tmp_path / "configs" / "resources.yaml"
        pcfg = tmp_path / "configs" / "config.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "gated_thing": {
                    "url": "http://example.invalid/x.txt",
                    "local_path": "data/x.txt",
                    "optional": True,
                    "enabled_by": "drug_enrichment.enable_expression_enrichment",
                    "description": "gated",
                },
            },
        })
        _write_pipeline_yaml(pcfg, {
            "drug_enrichment": {},  # the flag is absent entirely
        })
        status = setup_resources(
            rcfg, target_dir=tmp_path, pipeline_config=pcfg,
        )
        assert status["gated_thing"] == "gated_skipped"


class TestDSigDBPostprocessorRegistration:
    """Postprocess token-vs-function registration."""

    def test_token_is_registered(self) -> None:
        # The exact YAML token used by configs/resources.yaml must be
        # callable; a token mismatch would crash with "Unknown
        # postprocess handler" at runtime.
        assert "extract_dsigdb_d3" in _POSTPROCESSORS
        assert callable(_POSTPROCESSORS["extract_dsigdb_d3"])

    def test_function_name_mirrors_token(self) -> None:
        # Convention check (matches _postprocess_plink_zip ↔
        # "extract_plink_zip" and _postprocess_repurposing_hub ↔
        # "repurposing_hub_tsv_to_csv").  Helps grep-ability when
        # debugging postprocess failures from logs.
        assert hasattr(resources_mod, "_postprocess_extract_dsigdb_d3")
        assert (
            _POSTPROCESSORS["extract_dsigdb_d3"]
            is resources_mod._postprocess_extract_dsigdb_d3
        )

    def test_extracts_d3_member_from_zip(self, tmp_path: Path) -> None:
        # Fabricate a DSigDB-shaped zip with a D3 member and verify it
        # ends up at final_path.  The handler should also clean up the
        # raw zip.
        raw = tmp_path / "DSigDB_All_raw.txt"  # default raw path
        d3_payload = (
            "Tamoxifen_Cell_GSE1\tDescription\tESR1\tPGR\tBRCA1\n"
            "Aspirin_Cell_GSE2\tDescription\tPTGS1\tPTGS2\n"
        )
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("DSigDB_D3.txt", d3_payload)
            zf.writestr("DSigDB_D1_unused.txt", "ignore me\n")

        final = tmp_path / "drug_perturbations" / "dsigdb_d3.tsv"
        _POSTPROCESSORS["extract_dsigdb_d3"](raw, final)

        assert final.exists()
        assert final.read_text() == d3_payload
        assert not raw.exists(), "raw zip should be cleaned up"

    def test_raises_when_no_d3_member(self, tmp_path: Path) -> None:
        raw = tmp_path / "no_d3_raw.txt"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("DSigDB_D1.txt", "no D3 here\n")

        final = tmp_path / "out.tsv"
        with pytest.raises(FileNotFoundError, match="D3"):
            _POSTPROCESSORS["extract_dsigdb_d3"](raw, final)

    def test_raises_when_not_a_zip(self, tmp_path: Path) -> None:
        raw = tmp_path / "not_a_zip.txt"
        raw.write_text("hello not a zip")
        final = tmp_path / "out.tsv"
        with pytest.raises(ValueError, match="ZIP"):
            _POSTPROCESSORS["extract_dsigdb_d3"](raw, final)


class TestDsigdbManifestContract:
    """Regression guard: lock the configs/resources.yaml entry for ``dsigdb_d3``.

    Background: on 2026-05-18 the upstream DSigDB project removed the
    ``DSigDB_All.zip`` archive and migrated to direct per-signature GMT
    downloads. The manifest entry was updated to fetch ``D3.gmt`` directly
    (no postprocess required), since ``load_dsigdb`` is GMT-tolerant by
    construction. This test ensures the manifest never silently reverts to
    the dead ZIP URL (the entry would still validate against the YAML
    schema but every download would 404, and ``setup_resources`` reports
    that as a non-fatal "issue" alongside its "29/30 OK" success summary -
    easy to miss in CI).
    """

    _MANIFEST_PATH = Path(__file__).resolve().parents[1] / "configs" / "resources.yaml"

    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        with self._MANIFEST_PATH.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @pytest.fixture(scope="class")
    def dsigdb_entry(self, manifest: dict) -> dict:
        entry = manifest.get("resources", {}).get("dsigdb_d3")
        assert entry is not None, (
            "dsigdb_d3 resource entry missing from configs/resources.yaml - "
            "expression-enrichment mode-ON cannot acquire DSigDB without it."
        )
        return entry

    def test_url_points_at_direct_d3_gmt(self, dsigdb_entry: dict) -> None:
        url = dsigdb_entry.get("url", "")
        assert url.endswith("/D3.gmt"), (
            f"dsigdb_d3.url must end with '/D3.gmt' (direct GMT endpoint); "
            f"got {url!r}. The historic DSigDB_All.zip endpoint is dead (404)."
        )

    def test_url_uses_https(self, dsigdb_entry: dict) -> None:
        url = dsigdb_entry.get("url", "")
        assert url.startswith("https://"), (
            f"dsigdb_d3.url must use HTTPS for transport security; got {url!r}."
        )

    def test_no_postprocess_key(self, dsigdb_entry: dict) -> None:
        assert "postprocess" not in dsigdb_entry, (
            "dsigdb_d3 must not declare a postprocess for direct-GMT downloads; "
            "the historic 'extract_dsigdb_d3' postprocess only applies to the "
            "(now-dead) ZIP archive. Adding it back would re-introduce the "
            "ZIP-extraction failure mode for a non-ZIP payload."
        )

    def test_local_path_unchanged(self, dsigdb_entry: dict) -> None:
        # The Snakemake mode-ON DAG (workflows/rules/drug_enrichment.smk
        # line 66) hard-codes this exact path. Changing the extension or
        # directory would silently break load_drug_targets_expr.
        assert dsigdb_entry.get("local_path") == "resources/drug_perturbations/dsigdb_d3.tsv", (
            f"dsigdb_d3.local_path must remain "
            f"'resources/drug_perturbations/dsigdb_d3.tsv' for downstream "
            f"path-contract stability; got {dsigdb_entry.get('local_path')!r}."
        )

    def test_enabled_by_gate_preserved(self, dsigdb_entry: dict) -> None:
        assert (
            dsigdb_entry.get("enabled_by")
            == "drug_enrichment.enable_expression_enrichment"
        ), (
            "dsigdb_d3.enabled_by must remain "
            "'drug_enrichment.enable_expression_enrichment' for "
            "default-deny gating. Removing it would auto-download DSigDB on "
            "every setup-resources invocation, breaking mode-OFF semantics."
        )

    def test_optional_marker_preserved(self, dsigdb_entry: dict) -> None:
        assert dsigdb_entry.get("optional") is True, (
            "dsigdb_d3.optional must remain True (documentation marker; "
            "complementary to enabled_by:)."
        )


# ---------------------------------------------------------------------------
# GRCh38 gene-loc resource + postprocessor
# ---------------------------------------------------------------------------


class TestNcbi38GeneLocManifestContract:
    """Regression guard for the ``ncbi_gene_loc_grch38`` manifest entry.

    The Branch B S-PrediXcan MHC fix depends on this resource being a
    direct ZIP download from the official MAGMA SURF mirror, post-processed
    by ``extract_ncbi38_gene_loc``.  A silent revert to (e.g.) the dead
    CTG redirect would degrade Branch B back to the GRCh37 projection
    fallback - same situation as before R1.  This test locks the entry's
    URL host, checksum format, postprocess token, and the
    backwards-compatibility guarantee that the GRCh37 entry is preserved.
    """

    _MANIFEST_PATH = Path(__file__).resolve().parents[1] / "configs" / "resources.yaml"

    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        with self._MANIFEST_PATH.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @pytest.fixture(scope="class")
    def grch38_entry(self, manifest: dict) -> dict:
        entry = manifest.get("resources", {}).get("ncbi_gene_loc_grch38")
        assert entry is not None, (
            "ncbi_gene_loc_grch38 missing from configs/resources.yaml - "
            "Branch B S-PrediXcan cannot resolve GRCh38 MHC genes without it."
        )
        return entry

    def test_url_is_https(self, grch38_entry: dict) -> None:
        url = grch38_entry.get("url", "")
        assert url.startswith("https://"), (
            f"URL must be HTTPS for reproducible secure downloads; got {url!r}"
        )

    def test_url_points_at_surf_official_mirror(self, grch38_entry: dict) -> None:
        # The official MAGMA distribution page (https://cncr.nl/research/magma/)
        # links to the SURF Nextcloud-hosted ZIP at vu.data.surf.nl.  The
        # historic CTG URL (https://ctg.cncr.nl/software/MAGMA/aux_files/NCBI38.zip)
        # is dead and redirects to HTML - verified empirically R1 v4.
        url = grch38_entry.get("url", "")
        assert "vu.data.surf.nl" in url, (
            f"URL must point at the SURF mirror linked from the official "
            f"MAGMA page; got {url!r}. The legacy CTG URL is dead."
        )

    def test_checksum_is_sha256_of_raw_zip(self, grch38_entry: dict) -> None:
        # When ``postprocess`` is set, ``setup_resources`` verifies the
        # checksum against the *downloaded* archive, not the extracted
        # member.  The raw ZIP SHA256 below was captured at R1 v4 and
        # surfaces upstream drift loudly via ``checksum_mismatch``.
        checksum = grch38_entry.get("checksum", "")
        assert checksum == (
            "sha256:a6386efe10f1d9248c2768a598363696496c66dd2cdffee33a9c3005c726365c"
        ), (
            f"Checksum must lock to the verified raw NCBI38.zip SHA256; got {checksum!r}. "
            "If upstream republishes, update this value AND the extracted-file "
            "SHA256 documented in configs/resources.yaml + the drift-detection "
            "literal in _postprocess_extract_ncbi38_gene_loc."
        )

    def test_postprocess_token(self, grch38_entry: dict) -> None:
        assert grch38_entry.get("postprocess") == "extract_ncbi38_gene_loc", (
            "ncbi_gene_loc_grch38.postprocess must be 'extract_ncbi38_gene_loc' "
            "to unpack the ZIP at download time."
        )

    def test_local_path(self, grch38_entry: dict) -> None:
        assert grch38_entry.get("local_path") == "resources/reference/NCBI38.gene.loc", (
            "local_path must match the value wired through "
            "configs/reference.yaml::reference.gene_loc_file_grch38."
        )

    def test_grch37_entry_preserved(self, manifest: dict) -> None:
        # R1 must not break Branch A - the GRCh37 entry stays as-is.
        ncbi37 = manifest.get("resources", {}).get("ncbi_gene_loc")
        assert ncbi37 is not None, (
            "ncbi_gene_loc (GRCh37) missing - Branch A MAGMA would break."
        )
        assert ncbi37.get("manual") is True, (
            "ncbi_gene_loc must remain manual:true (Branch A behaviour preserved)."
        )
        assert (
            ncbi37.get("local_path") == "resources/reference/NCBI37.3.gene.loc"
        ), "Branch A path contract must be preserved."


class TestNcbi38GeneLocPostprocessor:
    """Unit tests for ``_postprocess_extract_ncbi38_gene_loc``."""

    _GENE_LOC_PAYLOAD = (
        "3105\t6\t29942532\t29945457\t+\tHLA-A\n"
        "3106\t6\t31268749\t31272136\t-\tHLA-B\n"
        "3107\t6\t31269491\t31357188\t-\tHLA-C\n"
    )

    def test_token_is_registered(self) -> None:
        assert "extract_ncbi38_gene_loc" in _POSTPROCESSORS
        assert callable(_POSTPROCESSORS["extract_ncbi38_gene_loc"])

    def test_function_name_mirrors_token(self) -> None:
        # Convention check, mirrors TestDSigDBPostprocessorRegistration.
        assert hasattr(resources_mod, "_postprocess_extract_ncbi38_gene_loc")
        assert (
            _POSTPROCESSORS["extract_ncbi38_gene_loc"]
            is resources_mod._postprocess_extract_ncbi38_gene_loc
        )

    def test_extracts_gene_loc_from_three_member_zip(self, tmp_path: Path) -> None:
        # Realistic MAGMA NCBI38.zip layout: REPORT + README + NCBI38.gene.loc.
        raw = tmp_path / "NCBI38_raw.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("REPORT", "MAGMA gene location summary\n")
            zf.writestr("README", "Source: NCBI gene_info, GRCh38\n")
            zf.writestr("NCBI38.gene.loc", self._GENE_LOC_PAYLOAD)

        final = tmp_path / "reference" / "NCBI38.gene.loc"
        _POSTPROCESSORS["extract_ncbi38_gene_loc"](raw, final)

        assert final.exists()
        assert final.read_text() == self._GENE_LOC_PAYLOAD
        assert not raw.exists(), "raw ZIP should be cleaned up after extraction"

    def test_raises_when_member_missing(self, tmp_path: Path) -> None:
        raw = tmp_path / "no_geneloc.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("REPORT", "irrelevant\n")
            zf.writestr("README", "no gene_loc here\n")

        final = tmp_path / "NCBI38.gene.loc"
        with pytest.raises(FileNotFoundError, match="NCBI38.gene.loc"):
            _POSTPROCESSORS["extract_ncbi38_gene_loc"](raw, final)

    def test_raises_when_not_a_zip(self, tmp_path: Path) -> None:
        # Pre-R1 the CTG URL returned HTML - guard against that recurring.
        raw = tmp_path / "html_page.zip"
        raw.write_text("<!DOCTYPE html><html><body>404</body></html>")
        final = tmp_path / "out.gene.loc"
        with pytest.raises(ValueError, match="ZIP"):
            _POSTPROCESSORS["extract_ncbi38_gene_loc"](raw, final)

    def test_rejects_path_traversal_member(self, tmp_path: Path) -> None:
        # A malicious ZIP with '../NCBI38.gene.loc' must not write outside
        # final_path's parent.  The handler should treat such a member as
        # invalid and surface a FileNotFoundError when no other matches.
        raw = tmp_path / "traversal.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("../NCBI38.gene.loc", self._GENE_LOC_PAYLOAD)

        final = tmp_path / "out" / "NCBI38.gene.loc"
        with pytest.raises(FileNotFoundError, match="NCBI38.gene.loc"):
            _POSTPROCESSORS["extract_ncbi38_gene_loc"](raw, final)

        sibling = tmp_path / "NCBI38.gene.loc"
        assert not sibling.exists(), (
            "Path traversal must not write a file outside final_path.parent"
        )

    def test_prefers_top_level_member_when_duplicated(self, tmp_path: Path) -> None:
        # If a ZIP contains both 'NCBI38.gene.loc' (top-level) and
        # 'extra/NCBI38.gene.loc' (subdirectory), pick the shorter path.
        raw = tmp_path / "dup.zip"
        top_payload = self._GENE_LOC_PAYLOAD
        nested_payload = "9999\t1\t1\t2\t+\tFAKE\n"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("extra/NCBI38.gene.loc", nested_payload)
            zf.writestr("NCBI38.gene.loc", top_payload)

        final = tmp_path / "NCBI38.gene.loc"
        _POSTPROCESSORS["extract_ncbi38_gene_loc"](raw, final)
        assert final.read_text() == top_payload


class TestMagmaBinaryPostprocessor:
    """Unit tests for ``_postprocess_extract_magma_binary``.

    MAGMA is fetched from its official distribution rather than conda: the
    conda-forge ``magma`` package is an unrelated GPU linear-algebra library,
    bioconda ships no equivalent, and the MAGMA licence forbids redistributing
    the binary inside a container image.
    """

    def test_token_is_registered(self) -> None:
        assert "extract_magma_binary" in _POSTPROCESSORS
        assert (
            _POSTPROCESSORS["extract_magma_binary"]
            is resources_mod._postprocess_extract_magma_binary
        )

    def test_extracts_and_marks_executable(self, tmp_path: Path) -> None:
        raw = tmp_path / "magma_raw.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("magma", "#!binary-placeholder\n")
            zf.writestr("README", "MAGMA v1.10\n")

        final = tmp_path / "bin" / "magma"
        _POSTPROCESSORS["extract_magma_binary"](raw, final)

        assert final.is_file()
        assert not raw.exists(), "raw ZIP should be cleaned up after extraction"
        # ZIP archives do not reliably carry POSIX permissions, so the execute
        # bit must be set explicitly or the binary cannot be run. Windows has
        # no POSIX permission bits, so this is asserted on POSIX only.
        if os.name != "nt":
            assert final.stat().st_mode & 0o100, "owner-execute bit must be set"

    def test_prefers_top_level_member(self, tmp_path: Path) -> None:
        raw = tmp_path / "nested.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("extras/aux/magma", "nested copy\n")
            zf.writestr("magma", "top-level copy\n")

        final = tmp_path / "bin" / "magma"
        _POSTPROCESSORS["extract_magma_binary"](raw, final)
        assert final.read_text() == "top-level copy\n"

    def test_raises_when_binary_missing(self, tmp_path: Path) -> None:
        raw = tmp_path / "no_binary.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("README", "documentation only\n")

        with pytest.raises(FileNotFoundError, match="magma"):
            _POSTPROCESSORS["extract_magma_binary"](raw, tmp_path / "bin" / "magma")

    def test_raises_when_not_a_zip(self, tmp_path: Path) -> None:
        # A dead download URL typically returns an HTML error page.
        raw = tmp_path / "html_page.zip"
        raw.write_text("<!DOCTYPE html><html><body>404</body></html>")

        with pytest.raises(ValueError, match="ZIP"):
            _POSTPROCESSORS["extract_magma_binary"](raw, tmp_path / "bin" / "magma")

    def test_rejects_path_traversal_member(self, tmp_path: Path) -> None:
        raw = tmp_path / "traversal.zip"
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("../magma", "escaped\n")

        with pytest.raises(FileNotFoundError, match="magma"):
            _POSTPROCESSORS["extract_magma_binary"](raw, tmp_path / "bin" / "magma")


class TestMagmaBinaryManifestContract:
    """Lock the MAGMA manifest entry so the download cannot silently drift."""

    def test_manifest_entry_is_wired(self) -> None:
        manifest = yaml.safe_load(
            Path("configs/resources.yaml").read_text(encoding="utf-8")
        )
        spec = manifest["resources"]["magma_binary"]
        assert spec["postprocess"] == "extract_magma_binary"
        assert spec["local_path"] == "resources/bin/magma"
        # Official VU/CNCR distribution; a change here needs a licence re-check.
        assert spec["url"].startswith("https://vu.data.surfsara.nl/")

    def test_conda_env_does_not_declare_magma(self) -> None:
        """`magma` on conda-forge is the wrong software - it must stay out."""
        env = yaml.safe_load(Path("envs/repogen.yaml").read_text(encoding="utf-8"))
        deps = [d for d in env["dependencies"] if isinstance(d, str)]
        assert not any(d == "magma" or d.startswith("magma=") or d.startswith("magma>")
                       for d in deps)


class TestEnvironmentLockContract:
    """Guard the reproducible-deployment artefacts.

    envs/repogen.yaml carries version ranges, so a fresh solve drifts over
    time. The lock files pin the exact package set RepoGen was validated
    with; these checks catch the ways a lock file silently stops working.
    """

    _LOCK = Path("envs/repogen.linux-64.lock")
    _PIP = Path("envs/repogen-pip.linux-64.txt")

    def test_lock_file_exists_and_is_explicit(self) -> None:
        text = self._LOCK.read_text(encoding="utf-8")
        assert "@EXPLICIT" in text
        assert "platform: linux-64" in text

    def test_lock_has_no_local_paths(self) -> None:
        """file:// entries resolve only on the machine that built the lock."""
        lines = [
            ln.strip() for ln in self._LOCK.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith(("#", "@"))
        ]
        assert lines, "lock file lists no packages"
        offenders = [ln for ln in lines if not ln.startswith("https://")]
        assert not offenders, f"non-remote package URLs in lock: {offenders[:3]}"

    def test_lock_pins_python_312(self) -> None:
        """Deployment targets 3.12 - the version the pipeline was validated on."""
        text = self._LOCK.read_text(encoding="utf-8")
        assert "/python-3.12" in text

    def test_lock_includes_external_binaries(self) -> None:
        """PLINK ships via conda; MAGMA deliberately does not (see manifest)."""
        text = self._LOCK.read_text(encoding="utf-8")
        assert "/plink-" in text
        assert "/snakemake-" in text

    def test_pip_requirements_exclude_repogen_itself(self) -> None:
        """repogen is installed from the repository, never pinned as a dependency."""
        lines = [
            ln.strip() for ln in self._PIP.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        assert lines, "pip requirements file lists no packages"
        assert all("==" in ln for ln in lines), "every pip package must be pinned"
        assert not any(ln.lower().startswith("repogen") for ln in lines)


class TestResourceUrlHygiene:
    """Catch download URLs that cannot succeed.

    A dead or mis-certificated URL only surfaces when someone runs
    setup-resources, which is typically their first experience of the
    pipeline. These checks pin the hosts that have already drifted once.
    """

    _MANIFEST = Path("configs/resources.yaml")

    def _resources(self) -> dict:
        return yaml.safe_load(self._MANIFEST.read_text(encoding="utf-8"))["resources"]

    def test_liftover_chain_uses_certificate_valid_host(self) -> None:
        """hgdownload.cse.ucsc.edu serves a certificate for a different name."""
        url = self._resources()["liftover_chain"]["url"]
        assert "hgdownload.soe.ucsc.edu" in url
        assert "hgdownload.cse.ucsc.edu" not in url

    def test_all_download_urls_are_https(self) -> None:
        offenders = [
            name for name, spec in self._resources().items()
            if (spec.get("url") or "").startswith("http://")
        ]
        assert not offenders, f"plaintext download URLs: {offenders}"

    def test_terms_gated_sources_stay_manual(self) -> None:
        """LINCS, eQTLGen, GTEx and PredictDB require the user to accept terms.

        Auto-downloading them would let a third party bypass acceptance, so
        they must keep manual: true even though direct URLs exist.
        """
        gated = ["lincs_l1000", "eqtlgen", "gtex_v8_eqtl", "predixcan_models"]
        resources = self._resources()
        for name in gated:
            assert resources[name].get("manual") is True, (
                f"{name} must remain a manual resource"
            )


class TestBranchAwareResourceSelection:
    """`setup-resources --branch a` must fetch Branch A's data and no more.

    The trap this guards against: Branch B's signature extraction and
    Branch C's Mendelian randomisation both consume drug_targets.parquet,
    which is built from ChEMBL/DGIdb/PDSP. Treating the drug databases as
    Branch A-only would produce a fetch that looks complete but leaves
    B and C unable to run.
    """

    _MANIFEST = Path("configs/resources.yaml")

    def _manifest(self) -> dict:
        return yaml.safe_load(self._MANIFEST.read_text(encoding="utf-8"))["resources"]

    def test_every_resource_declares_its_branches(self) -> None:
        untagged = [
            name for name, spec in self._manifest().items()
            if isinstance(spec, dict) and not spec.get("branches")
        ]
        assert not untagged, f"resources with no branches: tag: {untagged}"

    def test_branch_tags_are_valid(self) -> None:
        allowed = {"shared", "a", "b", "c"}
        for name, spec in self._manifest().items():
            tags = {str(t).lower() for t in spec.get("branches", [])}
            assert tags <= allowed, f"{name} has unknown branch tags: {tags - allowed}"

    def test_drug_sources_are_shared_not_branch_a_only(self) -> None:
        """Branches B and C need drug_targets.parquet, hence its inputs."""
        manifest = self._manifest()
        for name in ["chembl_sqlite", "dgidb_interactions", "pdsp_ki",
                     "chembl_pubchem_unichem"]:
            tags = {str(t).lower() for t in manifest[name]["branches"]}
            assert "shared" in tags, (
                f"{name} feeds drug_targets.parquet, which all branches read; "
                f"tagging it {tags} would break Branch B/C-only setups"
            )

    def test_gene_id_dictionaries_are_shared(self) -> None:
        manifest = self._manifest()
        for name in ["biomart_dico1", "biomart_dico2", "biomart_dico3",
                     "ncbi_gene_info", "ncbi_gene_history"]:
            assert "shared" in {str(t).lower() for t in manifest[name]["branches"]}

    def test_large_branch_specific_files_are_tagged_to_one_branch(self) -> None:
        """The 33 GB GCTX and the eQTL files must not be pulled by Branch A."""
        manifest = self._manifest()
        assert {str(t).lower() for t in manifest["lincs_l1000"]["branches"]} == {"b"}
        assert {str(t).lower() for t in manifest["eqtlgen"]["branches"]} == {"c"}

    def test_branch_a_selection_skips_lincs_and_eqtl(self, tmp_path: Path) -> None:
        rcfg = tmp_path / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "shared_thing": {"local_path": "s.txt", "branches": ["shared"],
                                 "manual": True, "description": "shared"},
                "branch_a_thing": {"local_path": "a.txt", "branches": ["a"],
                                   "manual": True, "description": "a"},
                "branch_b_thing": {"local_path": "b.txt", "branches": ["b"],
                                   "manual": True, "description": "b"},
                "branch_c_thing": {"local_path": "c.txt", "branches": ["c"],
                                   "manual": True, "description": "c"},
            },
        })
        status = setup_resources(rcfg, target_dir=tmp_path, branches={"a"})

        assert status["branch_b_thing"] == "skipped_other_branch"
        assert status["branch_c_thing"] == "skipped_other_branch"
        assert status["branch_a_thing"] != "skipped_other_branch"
        assert status["shared_thing"] != "skipped_other_branch", (
            "shared resources must be fetched for every branch selection"
        )

    def test_multiple_branches_union(self, tmp_path: Path) -> None:
        rcfg = tmp_path / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "a_thing": {"local_path": "a.txt", "branches": ["a"], "manual": True},
                "b_thing": {"local_path": "b.txt", "branches": ["b"], "manual": True},
                "c_thing": {"local_path": "c.txt", "branches": ["c"], "manual": True},
            },
        })
        status = setup_resources(rcfg, target_dir=tmp_path, branches={"a", "c"})
        assert status["a_thing"] != "skipped_other_branch"
        assert status["c_thing"] != "skipped_other_branch"
        assert status["b_thing"] == "skipped_other_branch"

    def test_no_branch_filter_processes_everything(self, tmp_path: Path) -> None:
        rcfg = tmp_path / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {
                "a_thing": {"local_path": "a.txt", "branches": ["a"], "manual": True},
                "b_thing": {"local_path": "b.txt", "branches": ["b"], "manual": True},
            },
        })
        status = setup_resources(rcfg, target_dir=tmp_path)
        assert "skipped_other_branch" not in status.values()

    def test_untagged_entry_is_treated_as_shared(self, tmp_path: Path) -> None:
        """An un-annotated manifest keeps its previous behaviour."""
        rcfg = tmp_path / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {"legacy": {"local_path": "x.txt", "manual": True}},
        })
        status = setup_resources(rcfg, target_dir=tmp_path, branches={"a"})
        assert status["legacy"] != "skipped_other_branch"

    def test_unknown_branch_is_rejected(self, tmp_path: Path) -> None:
        rcfg = tmp_path / "resources.yaml"
        _write_resources_yaml(rcfg, {
            "resources": {"x": {"local_path": "x.txt", "manual": True}},
        })
        with pytest.raises(ValueError, match="Unknown branch"):
            setup_resources(rcfg, target_dir=tmp_path, branches={"d"})
