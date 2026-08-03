"""Tests for repogen.analysis.drug_enrichment."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from statsmodels.stats.multitest import multipletests

from repogen.analysis.drug_enrichment import (
    assemble_drug_results,
    build_drug_gene_sets,
    collapse_pdsp_clusters,
    compute_permutation_enrichment,
    compute_wilcoxon_auc,
    create_drug_geneset_file,
    parse_magma_drug_results,
    run_drug_enrichment,
    run_magma_drug_enrichment,
)
from repogen.config.schema import DrugEnrichmentConfig, PipelineConfig, StudyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gene_results_df() -> pd.DataFrame:
    """Synthetic MAGMA gene results (~100 genes).

    Genes 1-10 have high Z-scores (signal), rest are noise.
    """
    rng = np.random.default_rng(42)
    n = 100
    z_scores = rng.normal(0, 1, n)
    z_scores[:10] = rng.normal(4, 0.5, 10)

    return pd.DataFrame({
        "gene_entrez_id": list(range(1, n + 1)),
        "gene_symbol": [f"GENE{i}" for i in range(1, n + 1)],
        "gene_ensembl_id": [f"ENSG{i:011d}" for i in range(1, n + 1)],
        "magma_z": z_scores,
        "magma_p": 1 - rng.uniform(0, 1, n),
        "n_snps": [50] * n,
        "chr": [1] * n,
        "start": list(range(1000, 1000 + n * 1000, 1000)),
        "end": list(range(2000, 2000 + n * 1000, 1000)),
        "in_mhc": [False] * n,
        "fdr_q": rng.uniform(0, 1, n),
    })


@pytest.fixture()
def drug_targets_df() -> pd.DataFrame:
    """Synthetic DrugTargetRecord with ~10 drugs.

    Drug A: targets high-Z genes (1-5) - should be enriched.
    Drug B: targets low-Z genes (80-85) - should NOT be enriched.
    Drug C: only 2 targets - excluded by min_genes=3.
    Drug D: mix of sources (chembl + pdsp).
    Drug E: all confidence="low".
    Drug F: pchembl < 6.0 threshold.
    Drug G-J: normal drugs with 4-6 targets each.
    """
    records = []

    for eid in [1, 2, 3, 4, 5]:
        records.append({
            "drug_chembl_id": "CHEMBL_A", "drug_name": "DrugA",
            "drug_inchikey": "AAAAAAAAAA-A", "drug_pubchem_cid": "CID100",
            "drug_smiles": "CCO", "gene_symbol": f"GENE{eid}",
            "gene_entrez_id": eid, "interaction_type": "antagonist",
            "pchembl_value": 8.5, "max_phase": 4, "atc_codes": ["N05A"],
            "mechanism_of_action": "dopamine antagonist",
            "indication_mesh": ["Depression"], "confidence": "high",
            "source": "chembl", "molecule_type": "small_molecule",
            "is_withdrawn": False, "source_pmids": ["12345678"],
        })

    for eid in [80, 81, 82, 83, 84, 85]:
        records.append({
            "drug_chembl_id": "CHEMBL_B", "drug_name": "DrugB",
            "drug_inchikey": "BBBBBBBBBB-B", "drug_pubchem_cid": "CID200",
            "drug_smiles": "CCCC", "gene_symbol": f"GENE{eid}",
            "gene_entrez_id": eid, "interaction_type": "inhibitor",
            "pchembl_value": 7.0, "max_phase": 3, "atc_codes": ["N06A"],
            "mechanism_of_action": "SSRI",
            "indication_mesh": ["Anxiety"], "confidence": "medium",
            "source": "chembl", "molecule_type": "small_molecule",
            "is_withdrawn": False, "source_pmids": ["23456789"],
        })

    for eid in [15, 16]:
        records.append({
            "drug_chembl_id": "CHEMBL_C", "drug_name": "DrugC",
            "drug_inchikey": None, "drug_pubchem_cid": None,
            "drug_smiles": None, "gene_symbol": f"GENE{eid}",
            "gene_entrez_id": eid, "interaction_type": "agonist",
            "pchembl_value": 6.0, "max_phase": 1, "atc_codes": [],
            "mechanism_of_action": None,
            "indication_mesh": [], "confidence": "medium",
            "source": "chembl", "molecule_type": "small_molecule",
            "is_withdrawn": False, "source_pmids": [],
        })

    for eid in [20, 21, 22, 23]:
        records.append({
            "drug_chembl_id": "CHEMBL_D", "drug_name": "DrugD",
            "drug_inchikey": "DDDDDDDDDD-D", "drug_pubchem_cid": "CID400",
            "drug_smiles": "C1CC1", "gene_symbol": f"GENE{eid}",
            "gene_entrez_id": eid, "interaction_type": "modulator",
            "pchembl_value": 7.5, "max_phase": 2, "atc_codes": ["N05B"],
            "mechanism_of_action": "GABA modulator",
            "indication_mesh": [], "confidence": "high",
            "source": "chembl" if eid < 22 else "pdsp",
            "molecule_type": "small_molecule",
            "is_withdrawn": False, "source_pmids": [],
        })

    for eid in [30, 31, 32, 33]:
        records.append({
            "drug_chembl_id": "CHEMBL_E", "drug_name": "DrugE",
            "drug_inchikey": None, "drug_pubchem_cid": None,
            "drug_smiles": None, "gene_symbol": f"GENE{eid}",
            "gene_entrez_id": eid, "interaction_type": "blocker",
            "pchembl_value": 5.5, "max_phase": 0, "atc_codes": [],
            "mechanism_of_action": None,
            "indication_mesh": [], "confidence": "low",
            "source": "dgidb", "molecule_type": None,
            "is_withdrawn": False, "source_pmids": [],
        })

    for eid in [40, 41, 42, 43]:
        records.append({
            "drug_chembl_id": "CHEMBL_F", "drug_name": "DrugF",
            "drug_inchikey": None, "drug_pubchem_cid": None,
            "drug_smiles": None, "gene_symbol": f"GENE{eid}",
            "gene_entrez_id": eid, "interaction_type": "inhibitor",
            "pchembl_value": 4.5, "max_phase": 1, "atc_codes": [],
            "mechanism_of_action": None,
            "indication_mesh": [], "confidence": "medium",
            "source": "chembl", "molecule_type": "small_molecule",
            "is_withdrawn": False, "source_pmids": [],
        })

    for drug_idx, start_gene in [(7, 50), (8, 55), (9, 60), (10, 65)]:
        for eid in range(start_gene, start_gene + 4):
            records.append({
                "drug_chembl_id": f"CHEMBL_G{drug_idx}", "drug_name": f"DrugG{drug_idx}",
                "drug_inchikey": None, "drug_pubchem_cid": None,
                "drug_smiles": None, "gene_symbol": f"GENE{eid}",
                "gene_entrez_id": eid, "interaction_type": "inhibitor",
                "pchembl_value": 7.0, "max_phase": 2, "atc_codes": [],
                "mechanism_of_action": None,
                "indication_mesh": [], "confidence": "medium",
                "source": "chembl", "molecule_type": "small_molecule",
                "is_withdrawn": False, "source_pmids": [],
            })

    df = pd.DataFrame(records)
    return df


@pytest.fixture()
def drug_gene_sets(gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame) -> dict[str, list[int]]:
    """Pre-built drug gene sets from the test fixtures."""
    sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)
    return sets


@pytest.fixture()
def sample_gsa_out(tmp_path: Path) -> Path:
    """Write a synthetic MAGMA .gsa.out file for drug enrichment."""
    content = (
        "# MAGMA gene-set analysis\n"
        "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P\n"
        "CHEMBL_A  COMPETITIVE  5  0.80  0.35  0.15  0.0001\n"
        "CHEMBL_B  COMPETITIVE  6  -0.10  -0.05  0.12  0.80\n"
        "CHEMBL_D  COMPETITIVE  4  0.30  0.15  0.10  0.02\n"
        "CHEMBL_E  COMPETITIVE  4  0.05  0.02  0.11  0.40\n"
        "CHEMBL_F  COMPETITIVE  4  0.10  0.05  0.10  0.20\n"
        "CHEMBL_G7  COMPETITIVE  4  0.15  0.08  0.09  0.15\n"
        "CHEMBL_G8  COMPETITIVE  4  0.12  0.06  0.09  0.18\n"
        "CHEMBL_G9  COMPETITIVE  4  0.08  0.04  0.10  0.30\n"
        "CHEMBL_G10  COMPETITIVE  4  0.20  0.10  0.08  0.08\n"
    )
    path = tmp_path / "drug_test.gsa.out"
    path.write_text(content)
    return path


def _make_config(
    tmp_path: Path,
    **drug_kwargs,
) -> PipelineConfig:
    """Build a minimal PipelineConfig for testing."""
    gwas_path = tmp_path / "test.gwas.gz"
    gwas_path.touch()
    return PipelineConfig(
        study=StudyConfig(name="test_study", gwas_input=gwas_path),
        drug_enrichment=DrugEnrichmentConfig(**drug_kwargs),
    )


def _write_gsa_out(path: Path, drugs: list[tuple]) -> None:
    """Helper to write a .gsa.out file.

    drugs: list of (name, ngenes, beta, beta_std, se, p).
    """
    lines = ["# MAGMA output\n", "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P\n"]
    for name, ng, beta, beta_std, se, p in drugs:
        lines.append(f"{name}  COMPETITIVE  {ng}  {beta}  {beta_std}  {se}  {p}\n")
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Gene Set Construction (5 tests)
# ---------------------------------------------------------------------------


class TestBuildDrugGeneSets:

    def test_basic(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """Correct drug->gene mapping and count."""
        sets, stats = build_drug_gene_sets(drug_targets_df, gene_results_df)
        assert "CHEMBL_A" in sets
        assert len(sets["CHEMBL_A"]) == 5
        assert "CHEMBL_B" in sets
        assert len(sets["CHEMBL_B"]) == 6
        assert all(isinstance(eid, int) for eid in sets["CHEMBL_A"])

    def test_min_genes_filter(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """Drug C (2 genes) excluded by min_genes_per_drug=3."""
        sets, stats = build_drug_gene_sets(drug_targets_df, gene_results_df, min_genes_per_drug=3)
        assert "CHEMBL_C" not in sets
        assert stats["n_drugs_dropped_min_genes"] >= 1

    def test_pchembl_filter(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """Drug F (pchembl=4.5) excluded when min_pchembl=6.0."""
        sets, stats = build_drug_gene_sets(
            drug_targets_df, gene_results_df, min_pchembl=6.0, min_genes_per_drug=3,
        )
        assert "CHEMBL_F" not in sets
        assert "CHEMBL_A" in sets
        assert stats["n_pairs_dropped_pchembl"] > 0

    def test_confidence_filter_medium(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """Drug E (all confidence='low') excluded when confidence_filter='medium'."""
        sets, stats = build_drug_gene_sets(
            drug_targets_df, gene_results_df, confidence_filter="medium",
        )
        assert "CHEMBL_E" not in sets
        assert "CHEMBL_A" in sets
        assert "CHEMBL_D" in sets
        assert stats["n_pairs_dropped_confidence"] > 0

    def test_no_entrez_dropped(
        self, gene_results_df: pd.DataFrame,
    ) -> None:
        """Genes without Entrez IDs are dropped."""
        dt = pd.DataFrame({
            "drug_chembl_id": ["DRUG1"] * 5,
            "drug_name": ["D1"] * 5,
            "gene_symbol": ["G1", "G2", "G3", "G4", "G5"],
            "gene_entrez_id": [1, 2, np.nan, 4, np.nan],
            "pchembl_value": [7.0] * 5,
            "max_phase": [2] * 5,
            "confidence": ["high"] * 5,
            "source": ["chembl"] * 5,
        })
        sets, stats = build_drug_gene_sets(dt, gene_results_df, min_genes_per_drug=3)
        assert "DRUG1" in sets
        assert len(sets["DRUG1"]) == 3
        assert stats["n_pairs_dropped_no_entrez"] == 2

    def test_confidence_filter_high(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """confidence_filter='high' keeps only high-confidence interactions.

        Drug E (all 'low') should be excluded. Drug F (all 'medium') should also
        be excluded. Drug A and D (confidence='high') should remain.
        """
        sets, stats = build_drug_gene_sets(
            drug_targets_df, gene_results_df, confidence_filter="high",
        )
        assert "CHEMBL_E" not in sets
        assert "CHEMBL_F" not in sets
        assert "CHEMBL_A" in sets
        assert "CHEMBL_D" in sets
        assert stats["n_pairs_dropped_confidence"] > 0

    def test_all_entrez_null_returns_empty_with_stats(
        self, gene_results_df: pd.DataFrame,
    ) -> None:
        """All-null Entrez IDs produce empty sets; stats record the drop count."""
        dt = pd.DataFrame({
            "drug_chembl_id": ["DRUG1"] * 5,
            "drug_name": ["D1"] * 5,
            "gene_symbol": ["G1", "G2", "G3", "G4", "G5"],
            "gene_entrez_id": [pd.NA] * 5,
            "pchembl_value": [7.0] * 5,
            "max_phase": [2] * 5,
            "confidence": ["high"] * 5,
            "source": ["chembl"] * 5,
        })
        sets, stats = build_drug_gene_sets(dt, gene_results_df, min_genes_per_drug=1)
        assert len(sets) == 0
        assert stats["n_pairs_dropped_no_entrez"] == 5

    def test_phase_filter_global_drops_all(
        self, gene_results_df: pd.DataFrame,
    ) -> None:
        """scope=global drops phase-0 drugs regardless of source."""
        dt = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"] * 3 + ["PDSP1"] * 3,
            "drug_name": ["C1"] * 3 + ["P1"] * 3,
            "gene_symbol": [f"G{i}" for i in [1, 2, 3, 4, 5, 6]],
            "gene_entrez_id": [1, 2, 3, 4, 5, 6],
            "pchembl_value": [7.0] * 6,
            "max_phase": [2, 2, 2, 0, 0, 0],
            "confidence": ["high"] * 6,
            "source": ["chembl"] * 3 + ["pdsp"] * 3,
        })
        sets, stats = build_drug_gene_sets(
            dt, gene_results_df, min_genes_per_drug=1,
            max_phase_filter=1, phase_filter_scope="global",
        )
        assert "CHEMBL1" in sets
        assert "PDSP1" not in sets
        assert stats["n_drugs_dropped_phase"] == 1

    def test_phase_filter_chembl_only_preserves_pdsp(
        self, gene_results_df: pd.DataFrame,
    ) -> None:
        """scope=chembl_only exempts PDSP/DGIdb drugs from phase gating."""
        dt = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1"] * 3 + ["PDSP1"] * 3 + ["DGI1"] * 3,
            "drug_name": ["C1"] * 3 + ["P1"] * 3 + ["D1"] * 3,
            "gene_symbol": [f"G{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "gene_entrez_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "pchembl_value": [7.0] * 9,
            "max_phase": [2, 2, 2, 0, 0, 0, 0, 0, 0],
            "confidence": ["high"] * 9,
            "source": ["chembl"] * 3 + ["pdsp"] * 3 + ["dgidb"] * 3,
        })
        sets, stats = build_drug_gene_sets(
            dt, gene_results_df, min_genes_per_drug=1,
            max_phase_filter=1, phase_filter_scope="chembl_only",
        )
        assert "CHEMBL1" in sets
        assert "PDSP1" in sets
        assert "DGI1" in sets
        assert stats["n_drugs_dropped_phase"] == 0

    def test_phase_filter_chembl_only_still_drops_low_phase_chembl(
        self, gene_results_df: pd.DataFrame,
    ) -> None:
        """scope=chembl_only still drops ChEMBL drugs below the phase threshold."""
        dt = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL_HI"] * 3 + ["CHEMBL_LO"] * 3 + ["PDSP1"] * 3,
            "drug_name": ["H"] * 3 + ["L"] * 3 + ["P"] * 3,
            "gene_symbol": [f"G{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "gene_entrez_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "pchembl_value": [7.0] * 9,
            "max_phase": [4, 4, 4, 0, 0, 0, 0, 0, 0],
            "confidence": ["high"] * 9,
            "source": ["chembl"] * 6 + ["pdsp"] * 3,
        })
        sets, stats = build_drug_gene_sets(
            dt, gene_results_df, min_genes_per_drug=1,
            max_phase_filter=1, phase_filter_scope="chembl_only",
        )
        assert "CHEMBL_HI" in sets
        assert "CHEMBL_LO" not in sets
        assert "PDSP1" in sets
        assert stats["n_drugs_dropped_phase"] == 1

    def test_phase_filter_chembl_only_multi_source_drug(
        self, gene_results_df: pd.DataFrame,
    ) -> None:
        """A drug with source='chembl,dgidb' IS subject to phase gating."""
        dt = pd.DataFrame({
            "drug_chembl_id": ["MULTI1"] * 3 + ["PDSP1"] * 3,
            "drug_name": ["M"] * 3 + ["P"] * 3,
            "gene_symbol": [f"G{i}" for i in [1, 2, 3, 4, 5, 6]],
            "gene_entrez_id": [1, 2, 3, 4, 5, 6],
            "pchembl_value": [7.0] * 6,
            "max_phase": [0, 0, 0, 0, 0, 0],
            "confidence": ["high"] * 6,
            "source": ["chembl,dgidb"] * 3 + ["pdsp"] * 3,
        })
        sets, stats = build_drug_gene_sets(
            dt, gene_results_df, min_genes_per_drug=1,
            max_phase_filter=1, phase_filter_scope="chembl_only",
        )
        assert "MULTI1" not in sets
        assert "PDSP1" in sets
        assert stats["n_drugs_dropped_phase"] == 1

    def test_phase_filter_none_ignores_scope(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """When max_phase_filter is None, scope is irrelevant."""
        sets_global, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            max_phase_filter=None, phase_filter_scope="global",
        )
        sets_chembl, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            max_phase_filter=None, phase_filter_scope="chembl_only",
        )
        assert set(sets_global.keys()) == set(sets_chembl.keys())


# ---------------------------------------------------------------------------
# Gene-set File Creation (3 tests)
# ---------------------------------------------------------------------------


class TestCreateDrugGenesetFile:

    def test_format(self, tmp_path: Path) -> None:
        """Correct tab-delimited format with integer Entrez IDs."""
        sets = {"CHEMBL1": [100, 200, 300], "CHEMBL2": [400, 500]}
        path = create_drug_geneset_file(sets, tmp_path / "test.txt")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parts = line.split("\t")
            assert parts[0].startswith("CHEMBL")
            for eid in parts[1:]:
                assert eid.isdigit()

    def test_sorted(self, tmp_path: Path) -> None:
        """Drugs are sorted alphabetically."""
        sets = {"CHEMBL_Z": [1, 2, 3], "CHEMBL_A": [4, 5, 6], "CHEMBL_M": [7, 8, 9]}
        path = create_drug_geneset_file(sets, tmp_path / "test.txt")
        lines = path.read_text().strip().split("\n")
        drug_ids = [line.split("\t")[0] for line in lines]
        assert drug_ids == sorted(drug_ids)

    def test_empty_raises(self, tmp_path: Path) -> None:
        """Empty dict raises ValueError."""
        with pytest.raises(ValueError, match="No drug gene sets"):
            create_drug_geneset_file({}, tmp_path / "test.txt")


# ---------------------------------------------------------------------------
# MAGMA Subprocess (3 tests)
# ---------------------------------------------------------------------------


class TestRunMagmaDrugEnrichment:

    @staticmethod
    def _write_valid_geneset(path: Path) -> None:
        path.write_text("CHEMBL25\t1234\t5678\nCHEMBL50\t3456\t7890\n")

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Mock subprocess, verify correct command tokens."""
        gsa_out = tmp_path / "test_drug_enrichment.gsa.out"
        gsa_out.write_text("VARIABLE TYPE NGENES BETA BETA_STD SE P\n")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        raw = tmp_path / "test.genes.raw"
        raw.touch()
        geneset = tmp_path / "test_drug_genesets.txt"
        self._write_valid_geneset(geneset)
        prefix = tmp_path / "test_drug_enrichment"

        result = run_magma_drug_enrichment(
            raw, geneset, prefix, magma_binary=Path("/usr/bin/magma"),
        )

        cmd = mock_run.call_args[0][0]
        assert "--gene-results" in cmd
        assert "--set-annot" in cmd
        assert "col=2,1" not in cmd
        set_annot_idx = cmd.index("--set-annot")
        assert cmd[set_annot_idx + 2] == "--out"
        assert result == gsa_out

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_failure_raises(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Non-zero exit code raises RuntimeError."""
        geneset = tmp_path / "test.txt"
        self._write_valid_geneset(geneset)
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="MAGMA error"
        )
        raw = tmp_path / "test.genes.raw"
        raw.touch()
        with pytest.raises(RuntimeError, match="MAGMA"):
            run_magma_drug_enrichment(
                raw, geneset, tmp_path / "out", magma_binary=Path("/usr/bin/magma"),
            )

    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_missing_output_raises(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Success exit but no .gsa.out - FileNotFoundError."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        raw = tmp_path / "test.genes.raw"
        raw.touch()
        geneset = tmp_path / "test.txt"
        self._write_valid_geneset(geneset)
        with pytest.raises(FileNotFoundError, match="gsa.out"):
            run_magma_drug_enrichment(
                raw, geneset, tmp_path / "out", magma_binary=Path("/usr/bin/magma"),
            )


# ---------------------------------------------------------------------------
# Result Parsing (3 tests)
# ---------------------------------------------------------------------------


class TestParseMagmaDrugResults:

    def test_basic(self, sample_gsa_out: Path) -> None:
        """Correct column names and row count."""
        df = parse_magma_drug_results(sample_gsa_out)
        assert len(df) == 9
        assert set(df.columns) == {
            "drug_chembl_id", "n_genes_in_magma", "magma_beta",
            "magma_beta_se", "magma_z", "magma_p",
        }

    def test_magma_z_equals_beta_over_se(self, sample_gsa_out: Path) -> None:
        """magma_z must equal magma_beta / magma_beta_se."""
        df = parse_magma_drug_results(sample_gsa_out)
        expected = df["magma_beta"] / df["magma_beta_se"]
        pd.testing.assert_series_equal(
            df["magma_z"], expected, check_names=False, atol=1e-10,
        )

    def test_empty_raises(self, tmp_path: Path) -> None:
        """Header-only file raises ValueError."""
        path = tmp_path / "empty.gsa.out"
        path.write_text("# comment\nVARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P\n")
        with pytest.raises(ValueError, match="empty"):
            parse_magma_drug_results(path)

    def test_column_types(self, sample_gsa_out: Path) -> None:
        """Verify float types for numeric columns."""
        df = parse_magma_drug_results(sample_gsa_out)
        assert df["magma_p"].dtype == np.float64
        assert df["magma_beta"].dtype == np.float64
        assert df["magma_beta_se"].dtype == np.float64
        assert df["n_genes_in_magma"].dtype in (np.int32, np.int64)

    def test_full_name_used_when_truncated(self, tmp_path: Path) -> None:
        """Parser must use FULL_NAME as drug_chembl_id when VARIABLE is truncated."""
        content = (
            "# MAGMA gene-set analysis\n"
            "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P  FULL_NAME\n"
            "PDSP_VERY_LONG_DRUG_NAME_TRU  COMPETITIVE  5  0.80  0.35  0.15  0.001  PDSP_VERY_LONG_DRUG_NAME_TRUNCATED_HERE\n"
            "CHEMBL123  COMPETITIVE  6  -0.10  -0.05  0.12  0.80  CHEMBL123\n"
        )
        path = tmp_path / "trunc.gsa.out"
        path.write_text(content)
        df = parse_magma_drug_results(path)
        assert "PDSP_VERY_LONG_DRUG_NAME_TRUNCATED_HERE" in df["drug_chembl_id"].values
        assert "PDSP_VERY_LONG_DRUG_NAME_TRU" not in df["drug_chembl_id"].values

    def test_hash_in_id_not_treated_as_comment(self, tmp_path: Path) -> None:
        """IDs containing # (e.g. HTML entities) must not be truncated."""
        content = (
            "# MAGMA gene-set analysis\n"
            "VARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P  FULL_NAME\n"
            "PDSP_DRUG_WITH_&#8242;  COMPETITIVE  5  0.50  0.25  0.10  0.01  PDSP_DRUG_WITH_&#8242;_PRIME\n"
        )
        path = tmp_path / "hash.gsa.out"
        path.write_text(content)
        df = parse_magma_drug_results(path)
        assert len(df) == 1
        assert "PRIME" in df["drug_chembl_id"].iloc[0]

    def test_invalid_utf8_handled(self, tmp_path: Path) -> None:
        """Parser must not crash on invalid UTF-8 bytes in VARIABLE."""
        header = b"# comment\nVARIABLE  TYPE  NGENES  BETA  BETA_STD  SE  P  FULL_NAME\n"
        bad_row = b"PDSP_BAD\xc3\x28NAME  COMPETITIVE  5  0.50  0.25  0.10  0.01  PDSP_BADNAME_FULL\n"
        path = tmp_path / "bad_utf8.gsa.out"
        path.write_bytes(header + bad_row)
        df = parse_magma_drug_results(path)
        assert len(df) == 1
        assert df["drug_chembl_id"].iloc[0] == "PDSP_BADNAME_FULL"


# ---------------------------------------------------------------------------
# Wilcoxon AUC (4 tests)
# ---------------------------------------------------------------------------


class TestWilcoxonAUC:

    def test_enriched_drug(self, gene_results_df: pd.DataFrame) -> None:
        """Drug targeting high-Z genes -> AUC significantly > 0.5."""
        sets = {"DRUG_HI": [1, 2, 3, 4, 5]}
        result = compute_wilcoxon_auc(
            gene_results_df["magma_z"], sets, gene_results_df["gene_entrez_id"],
        )
        row = result[result["drug_chembl_id"] == "DRUG_HI"].iloc[0]
        assert row["wilcoxon_auc"] > 0.7
        assert row["wilcoxon_p"] < 0.05

    def test_non_enriched_drug(self, gene_results_df: pd.DataFrame) -> None:
        """Drug targeting low-Z genes -> AUC <= 0.5."""
        sets = {"DRUG_LO": [80, 81, 82, 83, 84]}
        result = compute_wilcoxon_auc(
            gene_results_df["magma_z"], sets, gene_results_df["gene_entrez_id"],
        )
        row = result[result["drug_chembl_id"] == "DRUG_LO"].iloc[0]
        assert row["wilcoxon_auc"] <= 0.6

    def test_auc_range(self, gene_results_df: pd.DataFrame) -> None:
        """All AUC values between 0.0 and 1.0."""
        sets = {
            "D1": [1, 2, 3, 4], "D2": [50, 51, 52, 53], "D3": [80, 81, 82, 83],
        }
        result = compute_wilcoxon_auc(
            gene_results_df["magma_z"], sets, gene_results_df["gene_entrez_id"],
        )
        assert (result["wilcoxon_auc"] >= 0.0).all()
        assert (result["wilcoxon_auc"] <= 1.0).all()

    def test_degenerate(self) -> None:
        """All target genes with identical Z -> AUC = 0.5."""
        z = pd.Series([2.0] * 5 + [1.0, 3.0, 0.5, -1.0, 2.5])
        entrez = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sets = {"DRUG_DEGEN": [1, 2, 3, 4, 5]}
        result = compute_wilcoxon_auc(z, sets, entrez)
        row = result[result["drug_chembl_id"] == "DRUG_DEGEN"].iloc[0]
        assert row["wilcoxon_auc"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Permutation Test (4 tests)
# ---------------------------------------------------------------------------


class TestPermutationEnrichment:

    def test_enriched(self, gene_results_df: pd.DataFrame) -> None:
        """Drug targeting high-Z genes -> low p-value."""
        sets = {"DRUG_HI": [1, 2, 3, 4, 5]}
        result = compute_permutation_enrichment(
            gene_results_df["magma_z"].values,
            sets,
            gene_results_df["gene_entrez_id"].values,
            n_permutations=10000,
            permutation_seed=42,
        )
        row = result[result["drug_chembl_id"] == "DRUG_HI"].iloc[0]
        assert row["permutation_p"] < 0.05

    def test_reproducible(self, gene_results_df: pd.DataFrame) -> None:
        """Same seed -> same p-values."""
        sets = {"D1": [1, 2, 3, 4], "D2": [50, 51, 52, 53]}
        r1 = compute_permutation_enrichment(
            gene_results_df["magma_z"].values,
            sets,
            gene_results_df["gene_entrez_id"].values,
            n_permutations=1000,
            permutation_seed=123,
        )
        r2 = compute_permutation_enrichment(
            gene_results_df["magma_z"].values,
            sets,
            gene_results_df["gene_entrez_id"].values,
            n_permutations=1000,
            permutation_seed=123,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_resolution(self, gene_results_df: pd.DataFrame) -> None:
        """p-value >= 1/(n_permutations+1)."""
        n_perm = 1000
        sets = {"D1": [1, 2, 3, 4, 5]}
        result = compute_permutation_enrichment(
            gene_results_df["magma_z"].values,
            sets,
            gene_results_df["gene_entrez_id"].values,
            n_permutations=n_perm,
            permutation_seed=42,
        )
        assert (result["permutation_p"] >= 1.0 / (n_perm + 1)).all()

    def test_fdr_applied(self, gene_results_df: pd.DataFrame) -> None:
        """permutation_fdr_q is present and >= permutation_p after FDR."""
        sets = {"D1": [1, 2, 3, 4], "D2": [50, 51, 52, 53], "D3": [80, 81, 82, 83]}
        perm_results = compute_permutation_enrichment(
            gene_results_df["magma_z"].values,
            sets,
            gene_results_df["gene_entrez_id"].values,
            n_permutations=1000,
            permutation_seed=42,
        )
        _, perm_q, _, _ = multipletests(
            perm_results["permutation_p"].values, method="fdr_bh"
        )
        perm_results["permutation_fdr_q"] = perm_q
        assert "permutation_fdr_q" in perm_results.columns
        assert (perm_results["permutation_fdr_q"] >= perm_results["permutation_p"] - 1e-10).all()


# ---------------------------------------------------------------------------
# FDR Correction (2 tests)
# ---------------------------------------------------------------------------


class TestFDRCorrection:

    def test_fdr_applied(self, sample_gsa_out: Path) -> None:
        """FDR q-values present and >= raw p-values."""
        df = parse_magma_drug_results(sample_gsa_out)
        _, qvalues, _, _ = multipletests(df["magma_p"].values, method="fdr_bh")
        df["magma_fdr_q"] = qvalues
        assert (df["magma_fdr_q"] >= df["magma_p"] - 1e-10).all()

    def test_fdr_matches_statsmodels(self, sample_gsa_out: Path) -> None:
        """Module's FDR output matches direct statsmodels call."""
        df = parse_magma_drug_results(sample_gsa_out)
        raw_p = df["magma_p"].values

        _, expected_q, _, _ = multipletests(raw_p, method="fdr_bh")

        df["magma_fdr_q"] = multipletests(raw_p, method="fdr_bh")[1]

        np.testing.assert_array_almost_equal(df["magma_fdr_q"].values, expected_q)
        assert (df["magma_fdr_q"].values >= raw_p - 1e-10).all()
        assert len(set(df["magma_fdr_q"].values)) > 1


# ---------------------------------------------------------------------------
# Result Assembly (3 tests)
# ---------------------------------------------------------------------------


class TestAssembleDrugResults:

    def test_metadata_complete(
        self,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """All expected output columns present."""
        magma_results = parse_magma_drug_results(sample_gsa_out)
        _, q = multipletests(magma_results["magma_p"].values, method="fdr_bh")[:2]
        magma_results["magma_fdr_q"] = q

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)

        result = assemble_drug_results(
            magma_results, drug_targets_df, gene_results_df, sets,
        )

        expected_cols = {
            "drug_chembl_id", "drug_name", "magma_beta", "magma_beta_se",
            "magma_z", "magma_p", "magma_fdr_q", "n_target_genes",
            "target_genes", "mean_pchembl", "mean_target_z",
            "drug_pubchem_cid", "drug_smiles",
        }
        assert expected_cols <= set(result.columns)

    def test_target_genes_detail(
        self,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """target_genes list contains correct per-gene detail."""
        magma_results = parse_magma_drug_results(sample_gsa_out)
        _, q = multipletests(magma_results["magma_p"].values, method="fdr_bh")[:2]
        magma_results["magma_fdr_q"] = q

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)

        result = assemble_drug_results(
            magma_results, drug_targets_df, gene_results_df, sets,
        )

        chembl_a = result[result["drug_chembl_id"] == "CHEMBL_A"]
        if len(chembl_a) > 0:
            genes = chembl_a.iloc[0]["target_genes"]
            assert isinstance(genes, list)
            assert len(genes) > 0
            first_gene = genes[0]
            assert "gene_symbol" in first_gene
            assert "gene_entrez_id" in first_gene
            assert "magma_z" in first_gene
            assert "interaction_type" in first_gene
            assert "pchembl_value" in first_gene
            assert "source_pmids" in first_gene

    def test_sorted_by_p(
        self,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """Results sorted by magma_p ascending."""
        magma_results = parse_magma_drug_results(sample_gsa_out)
        _, q = multipletests(magma_results["magma_p"].values, method="fdr_bh")[:2]
        magma_results["magma_fdr_q"] = q

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)

        result = assemble_drug_results(
            magma_results, drug_targets_df, gene_results_df, sets,
        )

        p_vals = result["magma_p"].values
        assert all(p_vals[i] <= p_vals[i + 1] for i in range(len(p_vals) - 1))

    def test_sources_no_comma_tokens(
        self,
        gene_results_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """No individual source token should contain commas."""
        targets = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL_A"] * 3,
            "drug_name": ["DrugA"] * 3,
            "gene_symbol": ["GENE1", "GENE2", "GENE3"],
            "gene_entrez_id": [1, 2, 3],
            "source": ["chembl", "chembl,pdsp", "pdsp"],
            "confidence": ["high", "high", "medium"],
            "interaction_type": ["antagonist"] * 3,
            "pchembl_value": [8.0, 7.0, 6.0],
            "max_phase": [4, 4, 0],
            "mechanism_of_action": ["test"] * 3,
        })
        magma_results = parse_magma_drug_results(sample_gsa_out)
        _, q = multipletests(magma_results["magma_p"].values, method="fdr_bh")[:2]
        magma_results["magma_fdr_q"] = q
        sets = {"CHEMBL_A": [1, 2, 3]}
        result = assemble_drug_results(
            magma_results, targets, gene_results_df, sets,
        )
        row = result[result["drug_chembl_id"] == "CHEMBL_A"]
        if len(row) > 0 and "sources" in row.columns:
            sources = row.iloc[0]["sources"]
            for token in sources:
                assert "," not in str(token), f"Comma in source token: {token}"
            assert set(sources) == {"chembl", "pdsp"}

    def test_ndarray_atc_codes_survive_assembly(
        self,
        gene_results_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """atc_codes stored as numpy.ndarray (Parquet round-trip) must not be dropped."""
        targets = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL_A"] * 3,
            "drug_name": ["DrugA"] * 3,
            "gene_symbol": ["GENE1", "GENE2", "GENE3"],
            "gene_entrez_id": [1, 2, 3],
            "source": ["chembl"] * 3,
            "interaction_type": ["antagonist"] * 3,
            "pchembl_value": [8.0, 7.0, 6.0],
            "max_phase": [4, 4, 4],
            "mechanism_of_action": ["test"] * 3,
            "atc_codes": [
                np.array(["N05AH01", "N05AH02"], dtype=object),
                np.array(["N05AH01"], dtype=object),
                np.array([], dtype=object),
            ],
        })
        magma_results = parse_magma_drug_results(sample_gsa_out)
        _, q = multipletests(magma_results["magma_p"].values, method="fdr_bh")[:2]
        magma_results["magma_fdr_q"] = q
        sets = {"CHEMBL_A": [1, 2, 3]}
        result = assemble_drug_results(
            magma_results, targets, gene_results_df, sets,
        )
        row = result[result["drug_chembl_id"] == "CHEMBL_A"]
        assert len(row) == 1
        atc = row.iloc[0]["atc_codes"]
        assert set(atc) == {"N05AH01", "N05AH02"}

    def test_ndarray_indication_mesh_survive_assembly(
        self,
        gene_results_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """indication_mesh stored as numpy.ndarray must not be dropped."""
        targets = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL_A"] * 2,
            "drug_name": ["DrugA"] * 2,
            "gene_symbol": ["GENE1", "GENE2"],
            "gene_entrez_id": [1, 2],
            "source": ["chembl"] * 2,
            "interaction_type": ["antagonist"] * 2,
            "pchembl_value": [8.0, 7.0],
            "max_phase": [4, 4],
            "mechanism_of_action": ["test"] * 2,
            "indication_mesh": [
                np.array(["Depression", "Anxiety"], dtype=object),
                np.array(["Depression"], dtype=object),
            ],
        })
        magma_results = parse_magma_drug_results(sample_gsa_out)
        _, q = multipletests(magma_results["magma_p"].values, method="fdr_bh")[:2]
        magma_results["magma_fdr_q"] = q
        sets = {"CHEMBL_A": [1, 2]}
        result = assemble_drug_results(
            magma_results, targets, gene_results_df, sets,
        )
        row = result[result["drug_chembl_id"] == "CHEMBL_A"]
        assert len(row) == 1
        mesh = row.iloc[0]["indication_mesh"]
        assert set(mesh) == {"Depression", "Anxiety"}


# ---------------------------------------------------------------------------
# End-to-End Orchestration (3 tests, MAGMA mocked)
# ---------------------------------------------------------------------------


class TestRunDrugEnrichment:

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_full_pipeline(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """Full pipeline with mocked MAGMA -> Parquet + JSON."""
        mock_detect.return_value = Path("/usr/bin/magma")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        de_dir = out_dir / "drug_enrichment"
        de_dir.mkdir(parents=True)

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)
        gsa_path = de_dir / "test_study_drug_enrichment.gsa.out"
        drugs_for_gsa = [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ]
        _write_gsa_out(gsa_path, drugs_for_gsa)

        config = DrugEnrichmentConfig()
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=config,
            output_dir=out_dir,
            study_name="test_study",
        )

        parquet_path = de_dir / "test_study_drug_enrichment.parquet"
        json_path = de_dir / "test_study_drug_enrichment_metadata.json"
        assert parquet_path.exists()
        assert json_path.exists()
        assert len(results) > 0
        assert "magma_fdr_q" in results.columns

    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_no_drugs_raises(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
    ) -> None:
        """All drugs filtered out -> ValueError."""
        mock_detect.return_value = Path("/usr/bin/magma")

        dt = pd.DataFrame({
            "drug_chembl_id": ["X1", "X1"],
            "drug_name": ["X", "X"],
            "gene_symbol": ["G1", "G2"],
            "gene_entrez_id": [9999, 9998],
            "pchembl_value": [7.0, 7.0],
            "max_phase": [2, 2],
            "confidence": ["high", "high"],
            "source": ["chembl", "chembl"],
        })

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        config = DrugEnrichmentConfig(min_genes_per_drug=3)
        with pytest.raises(ValueError, match="No drugs passed"):
            run_drug_enrichment(
                gene_results_raw=raw_file,
                gene_results_df=gene_results_df,
                drug_targets=dt,
                config=config,
                output_dir=tmp_path / "output",
                study_name="test_study",
            )

    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_zero_entrez_error_mentions_config(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
    ) -> None:
        """All-null Entrez IDs -> ValueError mentioning reference.ncbi_gene_info."""
        mock_detect.return_value = Path("/usr/bin/magma")

        dt = pd.DataFrame({
            "drug_chembl_id": ["DRUG1"] * 5,
            "drug_name": ["D1"] * 5,
            "gene_symbol": [f"G{i}" for i in range(5)],
            "gene_entrez_id": [pd.NA] * 5,
            "pchembl_value": [7.0] * 5,
            "max_phase": [2] * 5,
            "confidence": ["high"] * 5,
            "source": ["chembl"] * 5,
        })

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        config = DrugEnrichmentConfig(min_genes_per_drug=1)
        with pytest.raises(ValueError, match="reference.ncbi_gene_info"):
            run_drug_enrichment(
                gene_results_raw=raw_file,
                gene_results_df=gene_results_df,
                drug_targets=dt,
                config=config,
                output_dir=tmp_path / "output",
                study_name="test_study",
            )

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_metadata_json(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """JSON metadata contains all expected fields."""
        mock_detect.return_value = Path("/usr/bin/magma")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        de_dir = out_dir / "drug_enrichment"
        de_dir.mkdir(parents=True)

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)
        gsa_path = de_dir / "test_study_drug_enrichment.gsa.out"
        drugs_for_gsa = [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ]
        _write_gsa_out(gsa_path, drugs_for_gsa)

        config = DrugEnrichmentConfig()
        run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=config,
            output_dir=out_dir,
            study_name="test_study",
        )

        json_path = de_dir / "test_study_drug_enrichment_metadata.json"
        with open(json_path) as f:
            meta = json.load(f)

        assert meta["result_type"] == "drug_enrichment"
        assert meta["study"] == "test_study"
        assert "timestamp" in meta
        assert "parameters" in meta
        assert "summary" in meta
        assert meta["parameters"]["fdr_method"] == "fdr_bh"
        assert meta["summary"]["n_drugs_tested"] > 0
        assert "n_drugs_dropped_min_genes" in meta["summary"]
        assert "n_pairs_dropped_no_entrez" in meta["summary"]
        assert "n_drugs_excluded_min_genes" not in meta["summary"]

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_wilcoxon_disabled(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """include_wilcoxon_auc=False -> no Wilcoxon columns in output."""
        mock_detect.return_value = Path("/usr/bin/magma")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        out_dir = tmp_path / "output"
        de_dir = out_dir / "drug_enrichment"
        de_dir.mkdir(parents=True)

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)
        gsa_path = de_dir / "test_study_drug_enrichment.gsa.out"
        drugs_for_gsa = [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ]
        _write_gsa_out(gsa_path, drugs_for_gsa)

        config = DrugEnrichmentConfig(include_wilcoxon_auc=False)
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=config,
            output_dir=out_dir,
            study_name="test_study",
        )

        assert "wilcoxon_auc" not in results.columns
        assert "wilcoxon_p" not in results.columns
        assert "magma_fdr_q" in results.columns
        assert "drug_chembl_id" in results.columns
        assert len(results) > 0

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_permutation_enabled(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """permutation_test=True -> permutation columns in final output."""
        mock_detect.return_value = Path("/usr/bin/magma")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        out_dir = tmp_path / "output"
        de_dir = out_dir / "drug_enrichment"
        de_dir.mkdir(parents=True)

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        sets, _stats = build_drug_gene_sets(drug_targets_df, gene_results_df)
        gsa_path = de_dir / "test_study_drug_enrichment.gsa.out"
        drugs_for_gsa = [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ]
        _write_gsa_out(gsa_path, drugs_for_gsa)

        config = DrugEnrichmentConfig(
            permutation_test=True, n_permutations=100, permutation_seed=42,
        )
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=config,
            output_dir=out_dir,
            study_name="test_study",
        )

        assert "permutation_p" in results.columns
        assert "permutation_fdr_q" in results.columns
        assert results["permutation_p"].notna().all()
        assert results["permutation_fdr_q"].notna().all()
        assert (results["permutation_p"] > 0).all()
        assert (results["permutation_p"] <= 1).all()
        assert (results["permutation_fdr_q"] >= results["permutation_p"] - 1e-10).all()

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_phase_filter_scope_passthrough(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
    ) -> None:
        """run_drug_enrichment honours phase_filter_scope from config.

        With chembl_only, PDSP drugs (phase 0) survive; with global, they are dropped.
        """
        mock_detect.return_value = Path("/usr/bin/magma")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        targets = pd.DataFrame({
            "drug_chembl_id": (
                ["CHEMBL_HI"] * 5 + ["PDSP_1"] * 5
            ),
            "drug_name": ["Hi"] * 5 + ["Pdsp1"] * 5,
            "gene_symbol": [f"GENE{i}" for i in [1, 2, 3, 4, 5, 80, 81, 82, 83, 84]],
            "gene_entrez_id": [1, 2, 3, 4, 5, 80, 81, 82, 83, 84],
            "pchembl_value": [8.0] * 10,
            "max_phase": [4] * 5 + [0] * 5,
            "confidence": ["high"] * 10,
            "source": ["chembl"] * 5 + ["pdsp"] * 5,
            "interaction_type": ["antagonist"] * 10,
            "mechanism_of_action": ["test"] * 10,
        })

        def _run_with_scope(scope: str) -> pd.DataFrame:
            out = tmp_path / f"out_{scope}"
            de_dir = out / "drug_enrichment"
            de_dir.mkdir(parents=True)

            raw_file = tmp_path / f"raw_{scope}.genes.raw"
            raw_file.touch()

            sets, _ = build_drug_gene_sets(
                targets, gene_results_df, min_genes_per_drug=3,
                max_phase_filter=1, phase_filter_scope=scope,
            )
            gsa_path = de_dir / f"test_{scope}_drug_enrichment.gsa.out"
            _write_gsa_out(gsa_path, [
                (did, len(genes), 0.3, 0.15, 0.1, 0.05)
                for did, genes in sorted(sets.items())
            ])

            cfg = DrugEnrichmentConfig(
                max_phase_filter=1, phase_filter_scope=scope,
                min_genes_per_drug=3,
            )
            return run_drug_enrichment(
                gene_results_raw=raw_file,
                gene_results_df=gene_results_df,
                drug_targets=targets,
                config=cfg,
                output_dir=out,
                study_name=f"test_{scope}",
            )

        results_global = _run_with_scope("global")
        assert "CHEMBL_HI" in results_global["drug_chembl_id"].values
        assert "PDSP_1" not in results_global["drug_chembl_id"].values

        results_chembl = _run_with_scope("chembl_only")
        assert "CHEMBL_HI" in results_chembl["drug_chembl_id"].values
        assert "PDSP_1" in results_chembl["drug_chembl_id"].values

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_pdsp_dedup_metadata(
        self,
        mock_run: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
    ) -> None:
        """pdsp_dedup_mode='exact_signature' updates metadata and reduces drug count."""
        mock_detect.return_value = Path("/usr/bin/magma")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        targets = pd.DataFrame({
            "drug_chembl_id": (
                ["CHEMBL_HI"] * 5
                + ["PDSP_A"] * 4 + ["PDSP_B"] * 4 + ["PDSP_C"] * 4
            ),
            "drug_name": (
                ["Hi"] * 5
                + ["PdspA"] * 4 + ["PdspB"] * 4 + ["PdspC"] * 4
            ),
            "gene_symbol": (
                [f"GENE{i}" for i in [1, 2, 3, 4, 5]]
                + [f"GENE{i}" for i in [80, 81, 82, 83]] * 3
            ),
            "gene_entrez_id": (
                [1, 2, 3, 4, 5]
                + [80, 81, 82, 83] * 3
            ),
            "pchembl_value": [8.0] * 17,
            "max_phase": [4] * 5 + [0] * 12,
            "confidence": ["high"] * 17,
            "source": ["chembl"] * 5 + ["pdsp"] * 12,
            "interaction_type": ["antagonist"] * 17,
            "mechanism_of_action": ["test"] * 17,
            "atc_codes": [["N05A"]] * 5 + [[]] * 12,
        })

        out = tmp_path / "out_dedup"
        de_dir = out / "drug_enrichment"
        de_dir.mkdir(parents=True)

        raw_file = tmp_path / "raw.genes.raw"
        raw_file.touch()

        sets, _ = build_drug_gene_sets(
            targets, gene_results_df, min_genes_per_drug=3,
        )
        gsa_path = de_dir / "test_dedup_drug_enrichment.gsa.out"
        _write_gsa_out(gsa_path, [
            (did, len(genes), 0.3, 0.15, 0.1, 0.05)
            for did, genes in sorted(sets.items())
            if did in {"CHEMBL_HI"} or not did.startswith("PDSP_") or did == "PDSP_A"
        ])

        cfg = DrugEnrichmentConfig(
            pdsp_dedup_mode="exact_signature",
            min_genes_per_drug=3,
        )
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=targets,
            config=cfg,
            output_dir=out,
            study_name="test_dedup",
        )

        json_path = de_dir / "test_dedup_drug_enrichment_metadata.json"
        with open(json_path) as f:
            meta = json.load(f)

        assert meta["parameters"]["pdsp_dedup_mode"] == "exact_signature"
        assert meta["summary"]["n_pdsp_drugs_collapsed"] == 2
        assert meta["summary"]["n_drugs_pre_dedup"] > meta["summary"]["n_drugs_tested"]

        sidecar = de_dir / "test_dedup_pdsp_clusters.json"
        assert sidecar.exists()
        with open(sidecar) as f:
            cluster_map = json.load(f)
        assert len(cluster_map) > 0


# ---------------------------------------------------------------------------
# Config Validation (3 tests)
# ---------------------------------------------------------------------------


class TestDrugEnrichmentConfig:

    def test_defaults(self) -> None:
        """Verify default values match spec."""
        c = DrugEnrichmentConfig()
        assert c.sources == ["chembl"]
        assert c.min_genes_per_drug == 3
        assert c.min_pchembl is None
        assert c.max_phase_filter is None
        assert c.phase_filter_scope == "global"
        assert c.confidence_filter is None
        assert c.pdsp_dedup_mode == "off"
        assert c.fdr_method == "fdr_bh"
        assert c.fdr_threshold == 0.05
        assert c.include_wilcoxon_auc is True
        assert c.permutation_test is False
        assert c.n_permutations == 10000
        assert c.permutation_seed == 42

    def test_invalid_fdr_method(self) -> None:
        """Invalid fdr_method raises ValidationError; fdr_tsbh accepted."""
        with pytest.raises(ValidationError, match="fdr_method"):
            DrugEnrichmentConfig(fdr_method="invalid")
        DrugEnrichmentConfig(fdr_method="fdr_tsbh")
        DrugEnrichmentConfig(fdr_method="fdr_tsbky")

    def test_invalid_confidence_filter(self) -> None:
        """Invalid confidence_filter raises ValidationError."""
        with pytest.raises(ValidationError, match="confidence_filter"):
            DrugEnrichmentConfig(confidence_filter="very_high")

    def test_phase_filter_scope_valid(self) -> None:
        assert DrugEnrichmentConfig(phase_filter_scope="global").phase_filter_scope == "global"
        assert DrugEnrichmentConfig(phase_filter_scope="chembl_only").phase_filter_scope == "chembl_only"

    def test_phase_filter_scope_invalid(self) -> None:
        with pytest.raises(ValidationError):
            DrugEnrichmentConfig(phase_filter_scope="per_source")

    def test_pdsp_dedup_mode_valid(self) -> None:
        assert DrugEnrichmentConfig(pdsp_dedup_mode="off").pdsp_dedup_mode == "off"
        assert DrugEnrichmentConfig(pdsp_dedup_mode="exact_signature").pdsp_dedup_mode == "exact_signature"

    def test_pdsp_dedup_mode_invalid(self) -> None:
        with pytest.raises(ValidationError):
            DrugEnrichmentConfig(pdsp_dedup_mode="family_overlap")
        with pytest.raises(ValidationError):
            DrugEnrichmentConfig(pdsp_dedup_mode="yes")

    def test_atc_min_genes_default_none(self) -> None:
        """Schema default is None (resolved via validator)."""
        c = DrugEnrichmentConfig()
        assert c.atc_min_genes_per_drug == 3

    def test_atc_min_genes_inherits_when_omitted(self) -> None:
        """Omitting atc_min_genes_per_drug inherits headline."""
        for n in (1, 2, 3, 5):
            c = DrugEnrichmentConfig(min_genes_per_drug=n)
            assert c.atc_min_genes_per_drug == n, (
                f"min_genes_per_drug={n} should inherit when omitted"
            )

    def test_atc_min_genes_explicit_lower(self) -> None:
        """Explicit atc < headline is permitted."""
        c = DrugEnrichmentConfig(min_genes_per_drug=3, atc_min_genes_per_drug=1)
        assert c.atc_min_genes_per_drug == 1

    def test_atc_min_genes_explicit_equal(self) -> None:
        """atc == headline is permitted (no-op for ATC pool)."""
        c = DrugEnrichmentConfig(min_genes_per_drug=3, atc_min_genes_per_drug=3)
        assert c.atc_min_genes_per_drug == 3

    def test_atc_min_genes_gt_headline_rejected(self) -> None:
        """Explicit atc > headline raises validation error."""
        with pytest.raises(ValidationError, match="atc_min_genes_per_drug"):
            DrugEnrichmentConfig(min_genes_per_drug=3, atc_min_genes_per_drug=4)

    def test_atc_min_genes_field_minimum_one(self) -> None:
        """Field-level constraint ge=1."""
        with pytest.raises(ValidationError):
            DrugEnrichmentConfig(atc_min_genes_per_drug=0)


# ---------------------------------------------------------------------------
# PDSP Cluster Collapse Tests
# ---------------------------------------------------------------------------


class TestCollapsePdspClusters:
    """Tests for collapse_pdsp_clusters()."""

    def _pdsp_targets(self, drug_id: str, entrez_ids: list[int],
                      pchembl: float = 7.0, atc: list | None = None,
                      source: str = "pdsp") -> list[dict]:
        """Helper: build drug_targets rows for one drug."""
        rows = []
        for eid in entrez_ids:
            rows.append({
                "drug_chembl_id": drug_id, "drug_name": drug_id,
                "gene_symbol": f"GENE{eid}", "gene_entrez_id": eid,
                "pchembl_value": pchembl, "source": source,
                "atc_codes": atc or [], "max_phase": 0,
                "confidence": "medium",
            })
        return rows

    def test_no_pdsp_drugs_noop(self) -> None:
        """When no PDSP-only drugs exist, nothing is collapsed."""
        gene_sets = {"CHEMBL1": [1, 2, 3], "CHEMBL2": [4, 5, 6]}
        dt = pd.DataFrame([
            {"drug_chembl_id": "CHEMBL1", "source": "chembl", "gene_entrez_id": 1,
             "pchembl_value": 7.0, "atc_codes": []},
            {"drug_chembl_id": "CHEMBL2", "source": "dgidb", "gene_entrez_id": 4,
             "pchembl_value": 6.0, "atc_codes": []},
        ])
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert set(result.keys()) == {"CHEMBL1", "CHEMBL2"}
        assert stats["n_pdsp_drugs_collapsed"] == 0

    def test_exact_signature_collapse(self) -> None:
        """Three PDSP-only drugs with identical targets -> 1 representative."""
        rows = (
            self._pdsp_targets("PDSP_ALPHA", [1, 2, 3])
            + self._pdsp_targets("PDSP_BETA", [1, 2, 3])
            + self._pdsp_targets("PDSP_GAMMA", [1, 2, 3])
        )
        dt = pd.DataFrame(rows)
        gene_sets = {
            "PDSP_ALPHA": [1, 2, 3],
            "PDSP_BETA": [1, 2, 3],
            "PDSP_GAMMA": [1, 2, 3],
        }
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert len(result) == 1
        assert stats["n_pdsp_drugs_collapsed"] == 2
        assert stats["n_pdsp_clusters"] == 1

    def test_mixed_source_preserved(self) -> None:
        """Drug with source='chembl,pdsp' must not be collapsed."""
        rows = (
            self._pdsp_targets("PDSP_A", [1, 2, 3])
            + self._pdsp_targets("MIXED", [1, 2, 3], source="chembl,pdsp")
        )
        dt = pd.DataFrame(rows)
        gene_sets = {"PDSP_A": [1, 2, 3], "MIXED": [1, 2, 3]}
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "MIXED" in result
        assert "PDSP_A" in result
        assert stats["n_pdsp_drugs_collapsed"] == 0

    def test_non_pdsp_untouched(self) -> None:
        """ChEMBL and DGIdb drugs with identical targets are not collapsed."""
        rows = (
            self._pdsp_targets("CHEMBL1", [1, 2, 3], source="chembl")
            + self._pdsp_targets("CHEMBL2", [1, 2, 3], source="chembl")
            + self._pdsp_targets("PDSP_X", [1, 2, 3])
            + self._pdsp_targets("PDSP_Y", [1, 2, 3])
        )
        dt = pd.DataFrame(rows)
        gene_sets = {
            "CHEMBL1": [1, 2, 3], "CHEMBL2": [1, 2, 3],
            "PDSP_X": [1, 2, 3], "PDSP_Y": [1, 2, 3],
        }
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "CHEMBL1" in result
        assert "CHEMBL2" in result
        assert stats["n_pdsp_drugs_collapsed"] == 1
        pdsp_kept = [k for k in result if k.startswith("PDSP_")]
        assert len(pdsp_kept) == 1

    def test_singleton_preserved(self) -> None:
        """PDSP-only drug with unique target set is never removed."""
        rows = (
            self._pdsp_targets("PDSP_LONE", [10, 20, 30])
            + self._pdsp_targets("PDSP_DUP1", [1, 2, 3])
            + self._pdsp_targets("PDSP_DUP2", [1, 2, 3])
        )
        dt = pd.DataFrame(rows)
        gene_sets = {
            "PDSP_LONE": [10, 20, 30],
            "PDSP_DUP1": [1, 2, 3],
            "PDSP_DUP2": [1, 2, 3],
        }
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "PDSP_LONE" in result
        assert stats["n_pdsp_drugs_collapsed"] == 1

    def test_representative_prefers_chembl_id(self) -> None:
        """Drug with real CHEMBL ID is preferred over PDSP_ placeholder."""
        rows = (
            self._pdsp_targets("PDSP_PLAIN", [1, 2, 3], pchembl=9.0)
            + self._pdsp_targets("CHEMBL999", [1, 2, 3], pchembl=5.0)
        )
        dt = pd.DataFrame(rows)
        gene_sets = {"PDSP_PLAIN": [1, 2, 3], "CHEMBL999": [1, 2, 3]}
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "CHEMBL999" in result
        assert "PDSP_PLAIN" not in result

    def test_representative_prefers_atc(self) -> None:
        """Among PDSP_ IDs, prefer the one with ATC codes."""
        rows = (
            self._pdsp_targets("PDSP_NO_ATC", [1, 2, 3], pchembl=9.0)
            + self._pdsp_targets("PDSP_HAS_ATC", [1, 2, 3], pchembl=5.0,
                                 atc=["N05A"])
        )
        dt = pd.DataFrame(rows)
        gene_sets = {"PDSP_NO_ATC": [1, 2, 3], "PDSP_HAS_ATC": [1, 2, 3]}
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "PDSP_HAS_ATC" in result
        assert "PDSP_NO_ATC" not in result

    def test_representative_prefers_higher_pchembl(self) -> None:
        """Among equal-priority drugs, higher mean pchembl wins."""
        rows = (
            self._pdsp_targets("PDSP_LOW", [1, 2, 3], pchembl=5.0)
            + self._pdsp_targets("PDSP_HIGH", [1, 2, 3], pchembl=9.0)
        )
        dt = pd.DataFrame(rows)
        gene_sets = {"PDSP_LOW": [1, 2, 3], "PDSP_HIGH": [1, 2, 3]}
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "PDSP_HIGH" in result
        assert "PDSP_LOW" not in result

    def test_representative_nan_pchembl_ranks_last(self) -> None:
        """Drug with NaN pchembl ranks below one with a real value."""
        rows = (
            self._pdsp_targets("PDSP_NAN", [1, 2, 3], pchembl=float("nan"))
            + self._pdsp_targets("PDSP_OK", [1, 2, 3], pchembl=6.0)
        )
        dt = pd.DataFrame(rows)
        gene_sets = {"PDSP_NAN": [1, 2, 3], "PDSP_OK": [1, 2, 3]}
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "PDSP_OK" in result
        assert "PDSP_NAN" not in result

    def test_lexical_tiebreak(self) -> None:
        """With all else equal, lexicographically first ID wins."""
        rows = (
            self._pdsp_targets("PDSP_ZZZ", [1, 2, 3])
            + self._pdsp_targets("PDSP_AAA", [1, 2, 3])
        )
        dt = pd.DataFrame(rows)
        gene_sets = {"PDSP_ZZZ": [1, 2, 3], "PDSP_AAA": [1, 2, 3]}
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert "PDSP_AAA" in result
        assert "PDSP_ZZZ" not in result

    def test_cluster_stats_correct(self) -> None:
        """Verify all returned stats keys and values."""
        rows = (
            self._pdsp_targets("PDSP_A1", [1, 2, 3])
            + self._pdsp_targets("PDSP_A2", [1, 2, 3])
            + self._pdsp_targets("PDSP_B1", [10, 20])
            + self._pdsp_targets("CHEMBL_X", [1, 2, 3], source="chembl")
        )
        dt = pd.DataFrame(rows)
        gene_sets = {
            "PDSP_A1": [1, 2, 3], "PDSP_A2": [1, 2, 3],
            "PDSP_B1": [10, 20], "CHEMBL_X": [1, 2, 3],
        }
        result, stats = collapse_pdsp_clusters(gene_sets, dt)
        assert stats["n_pdsp_only_candidates"] == 3
        assert stats["n_pdsp_clusters"] == 2
        assert stats["n_pdsp_drugs_collapsed"] == 1
        assert stats["n_drugs_pre_dedup"] == 4
        assert stats["n_drugs_post_dedup"] == 3
        assert len(stats["cluster_map"]) == 1


# ---------------------------------------------------------------------------
# Integration Test (1 test, skipped without MAGMA)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Two-tier min_genes_per_drug threshold (headline pool)
# ---------------------------------------------------------------------------


class TestHeadlineDrugPool:
    """Tests that exercise the inclusive ATC pool +
    headline flag + post-assemble FDR scoping + export filtering."""

    def test_build_drug_gene_sets_inherits_headline_when_atc_none(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """Omitting atc_min_genes_per_drug yields the same gate as the legacy
        single-threshold call - byte-identical behaviour for the 26 existing
        test sites that pre-date the two-tier threshold."""
        sets_legacy, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df, min_genes_per_drug=3,
        )
        sets_item7, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            min_genes_per_drug=3, atc_min_genes_per_drug=None,
        )
        assert set(sets_legacy.keys()) == set(sets_item7.keys())
        for k in sets_legacy:
            assert sets_legacy[k] == sets_item7[k]

    def test_build_drug_gene_sets_atc_lt_headline_includes_smaller_drugs(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """atc=1, headline=3 includes the 2-gene CHEMBL_C drug that the
        single-threshold legacy call would have excluded."""
        sets_legacy, stats_legacy = build_drug_gene_sets(
            drug_targets_df, gene_results_df, min_genes_per_drug=3,
        )
        sets_item7, stats_item7 = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            min_genes_per_drug=3, atc_min_genes_per_drug=1,
        )
        assert "CHEMBL_C" not in sets_legacy
        assert "CHEMBL_C" in sets_item7
        assert stats_item7["n_drugs_in_atc_pool"] >= stats_legacy["n_drugs_tested"]
        assert stats_item7["n_drugs_in_headline_pool"] == stats_legacy["n_drugs_tested"]
        assert stats_item7["atc_min_genes_per_drug"] == 1
        assert stats_item7["headline_min_genes_per_drug"] == 3

    def test_build_drug_gene_sets_filter_stats_keys(
        self, gene_results_df: pd.DataFrame, drug_targets_df: pd.DataFrame,
    ) -> None:
        """New filter_stats keys are present and consistent."""
        _, stats = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            min_genes_per_drug=3, atc_min_genes_per_drug=1,
        )
        for key in (
            "n_drugs_in_atc_pool", "n_drugs_in_headline_pool",
            "atc_min_genes_per_drug", "headline_min_genes_per_drug",
        ):
            assert key in stats
        assert stats["n_drugs_in_atc_pool"] >= stats["n_drugs_in_headline_pool"]

    def test_assemble_drug_results_default_min_genes_3(
        self,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """assemble_drug_results without min_genes_per_drug uses default 3
        - preserves API compatibility with 6 existing test sites."""
        magma_results = parse_magma_drug_results(sample_gsa_out)
        sets, _ = build_drug_gene_sets(drug_targets_df, gene_results_df)

        result = assemble_drug_results(
            magma_results, drug_targets_df, gene_results_df, sets,
        )

        assert "passes_headline_min_genes" in result.columns
        assert "n_target_genes_input" in result.columns
        assert result["passes_headline_min_genes"].dtype == bool
        for row in result.itertuples(index=False):
            expected = row.n_target_genes_input >= 3
            assert row.passes_headline_min_genes == expected

    def test_assemble_drug_results_uses_pre_magma_size(
        self,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """passes_headline_min_genes is gated on pre-MAGMA gene-set size,
        not post-MAGMA n_genes_in_magma. Critical for bit-identity when
        MAGMA drops genes with no SNPs in the reference panel."""
        sets = {"DRUG_X": [1, 2, 3], "DRUG_Y": [10]}
        magma_results = pd.DataFrame({
            "drug_chembl_id": ["DRUG_X", "DRUG_Y"],
            "n_genes_in_magma": [2, 1],
            "magma_beta": [0.3, 0.1],
            "magma_beta_se": [0.1, 0.1],
            "magma_z": [3.0, 1.0],
            "magma_p": [0.01, 0.30],
        })

        result = assemble_drug_results(
            magma_results, drug_targets_df.iloc[:0], gene_results_df, sets,
            min_genes_per_drug=3,
        )

        x_row = result[result["drug_chembl_id"] == "DRUG_X"].iloc[0]
        assert x_row["n_target_genes_input"] == 3
        assert x_row["n_target_genes"] == 2
        assert bool(x_row["passes_headline_min_genes"]) is True

        y_row = result[result["drug_chembl_id"] == "DRUG_Y"].iloc[0]
        assert y_row["n_target_genes_input"] == 1
        assert bool(y_row["passes_headline_min_genes"]) is False

    def test_assemble_drug_results_explicit_min_genes(
        self,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
        sample_gsa_out: Path,
    ) -> None:
        """min_genes_per_drug=1 sets the flag True for all drugs in the pool."""
        magma_results = parse_magma_drug_results(sample_gsa_out)
        sets, _ = build_drug_gene_sets(drug_targets_df, gene_results_df)

        result = assemble_drug_results(
            magma_results, drug_targets_df, gene_results_df, sets,
            min_genes_per_drug=1,
        )

        assert result["passes_headline_min_genes"].all()

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_run_drug_enrichment_baseline_byte_identical(
        self,
        mock_subproc: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """With atc_min_genes_per_drug omitted (= inherits 3), parquet shape
        and headline magma_fdr_q are bit-identical to the legacy run."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        mock_detect.return_value = Path("/usr/bin/magma")

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        for run_label, kwargs in (
            ("legacy", {"min_genes_per_drug": 3}),
            ("item7_inherit", {"min_genes_per_drug": 3, "atc_min_genes_per_drug": None}),
        ):
            de_dir = tmp_path / f"out_{run_label}" / "drug_enrichment"
            de_dir.mkdir(parents=True, exist_ok=True)
            sets, _ = build_drug_gene_sets(
                drug_targets_df, gene_results_df, min_genes_per_drug=3,
            )
            gsa_path = de_dir / f"test_{run_label}_drug_enrichment.gsa.out"
            _write_gsa_out(gsa_path, [
                (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
                for drug_id, genes in sorted(sets.items())
            ])

            cfg = DrugEnrichmentConfig(**kwargs)
            results = run_drug_enrichment(
                gene_results_raw=raw_file,
                gene_results_df=gene_results_df,
                drug_targets=drug_targets_df,
                config=cfg,
                output_dir=tmp_path / f"out_{run_label}",
                study_name=f"test_{run_label}",
            )
            globals()[f"_results_{run_label}"] = results

        legacy = globals()["_results_legacy"].sort_values("drug_chembl_id").reset_index(drop=True)
        item7 = globals()["_results_item7_inherit"].sort_values("drug_chembl_id").reset_index(drop=True)
        assert list(legacy["drug_chembl_id"]) == list(item7["drug_chembl_id"])
        assert legacy["passes_headline_min_genes"].all()
        assert item7["passes_headline_min_genes"].all()
        np.testing.assert_array_almost_equal(
            legacy["magma_fdr_q"].values, item7["magma_fdr_q"].values, decimal=12,
        )

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_run_drug_enrichment_inclusive_pool_with_subheadline(
        self,
        mock_subproc: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """atc_min_genes_per_drug=1, min_genes_per_drug=3:
        - parquet contains sub-headline drugs (e.g., CHEMBL_C with 2 genes)
        - sub-headline drugs have magma_fdr_q == NaN
        - headline drugs have FDR matching the legacy run (BH on subset)
        """
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        mock_detect.return_value = Path("/usr/bin/magma")

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        de_dir = tmp_path / "out" / "drug_enrichment"
        de_dir.mkdir(parents=True, exist_ok=True)
        sets, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            min_genes_per_drug=3, atc_min_genes_per_drug=1,
        )
        assert "CHEMBL_C" in sets
        gsa_path = de_dir / "test_drug_enrichment.gsa.out"
        _write_gsa_out(gsa_path, [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ])

        cfg = DrugEnrichmentConfig(min_genes_per_drug=3, atc_min_genes_per_drug=1)
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=cfg,
            output_dir=tmp_path / "out",
            study_name="test",
        )

        c_row = results[results["drug_chembl_id"] == "CHEMBL_C"]
        assert len(c_row) == 1
        assert bool(c_row.iloc[0]["passes_headline_min_genes"]) is False
        assert pd.isna(c_row.iloc[0]["magma_fdr_q"])

        headline = results[results["passes_headline_min_genes"]]
        assert headline["magma_fdr_q"].notna().all()

        # BH applied to the headline subset only - q-values bit-identical to
        # what one would get running multipletests on just those rows.
        _, expected_q, _, _ = multipletests(
            headline["magma_p"].values, method=cfg.fdr_method,
        )
        np.testing.assert_array_almost_equal(
            np.sort(headline["magma_fdr_q"].values),
            np.sort(expected_q),
            decimal=12,
        )

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_metadata_records_both_thresholds_and_pools(
        self,
        mock_subproc: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """metadata.parameters and metadata.summary record the new keys."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        mock_detect.return_value = Path("/usr/bin/magma")

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        de_dir = tmp_path / "out" / "drug_enrichment"
        de_dir.mkdir(parents=True, exist_ok=True)
        sets, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            min_genes_per_drug=3, atc_min_genes_per_drug=1,
        )
        gsa_path = de_dir / "meta_test_drug_enrichment.gsa.out"
        _write_gsa_out(gsa_path, [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ])

        cfg = DrugEnrichmentConfig(min_genes_per_drug=3, atc_min_genes_per_drug=1)
        run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=cfg,
            output_dir=tmp_path / "out",
            study_name="meta_test",
        )

        meta_path = de_dir / "meta_test_drug_enrichment_metadata.json"
        meta = json.loads(meta_path.read_text())

        assert meta["parameters"]["min_genes_per_drug"] == 3
        assert meta["parameters"]["atc_min_genes_per_drug"] == 1
        assert "n_drugs_in_atc_pool" in meta["summary"]
        assert "n_drugs_in_headline_pool" in meta["summary"]
        assert (
            meta["summary"]["n_drugs_in_atc_pool"]
            >= meta["summary"]["n_drugs_in_headline_pool"]
        )

    @patch("repogen.analysis.drug_enrichment._get_magma_version", return_value="v1.10")
    @patch("repogen.analysis.drug_enrichment.detect_magma_binary")
    @patch("repogen.analysis.magma_pathway.subprocess.run")
    def test_permutation_fdr_scoped_to_headline(
        self,
        mock_subproc: MagicMock,
        mock_detect: MagicMock,
        mock_version: MagicMock,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """When permutation_test=True, permutation_fdr_q is NaN for sub-headline
        drugs and computed via BH on the headline subset only."""
        mock_subproc.return_value = MagicMock(returncode=0, stderr="")
        mock_detect.return_value = Path("/usr/bin/magma")

        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        de_dir = tmp_path / "out" / "drug_enrichment"
        de_dir.mkdir(parents=True, exist_ok=True)
        sets, _ = build_drug_gene_sets(
            drug_targets_df, gene_results_df,
            min_genes_per_drug=3, atc_min_genes_per_drug=1,
        )
        gsa_path = de_dir / "perm_test_drug_enrichment.gsa.out"
        _write_gsa_out(gsa_path, [
            (drug_id, len(genes), 0.3, 0.15, 0.1, 0.05)
            for drug_id, genes in sorted(sets.items())
        ])

        cfg = DrugEnrichmentConfig(
            min_genes_per_drug=3, atc_min_genes_per_drug=1,
            permutation_test=True, n_permutations=200, permutation_seed=42,
        )
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=cfg,
            output_dir=tmp_path / "out",
            study_name="perm_test",
        )

        assert "permutation_fdr_q" in results.columns
        sub = results[~results["passes_headline_min_genes"]]
        if len(sub) > 0:
            assert sub["permutation_fdr_q"].isna().all()
        head = results[results["passes_headline_min_genes"]]
        assert head["permutation_fdr_q"].notna().all()


# ---------------------------------------------------------------------------
# Export-side filtering of sub-headline drugs
# ---------------------------------------------------------------------------


class TestHeadlineExportFiltering:
    """Tests for the export-side filtering helpers."""

    def _make_de_df(self) -> pd.DataFrame:
        """Synthetic drug enrichment DataFrame with mixed headline status."""
        return pd.DataFrame({
            "drug_chembl_id": ["A", "B", "C", "D"],
            "drug_name": ["A", "B", "C", "D"],
            "magma_p": [0.001, 0.01, 0.50, 0.80],
            "magma_fdr_q": [0.004, 0.03, np.nan, np.nan],
            "max_phase": [4, 3, 1, 1],
            "n_target_genes": [5, 4, 2, 1],
            "n_target_genes_input": [5, 4, 2, 1],
            "passes_headline_min_genes": [True, True, False, False],
            "atc_codes": [["N06A"], ["N06A"], [], []],
        })

    def test_filter_to_headline_drops_sub_headline(self) -> None:
        from repogen.reporting.export import _filter_to_headline
        df = self._make_de_df()
        out = _filter_to_headline(df)
        assert list(out["drug_chembl_id"]) == ["A", "B"]

    def test_filter_to_headline_idempotent_on_legacy_parquet(self) -> None:
        """Legacy parquet (pre-Item-7) lacks the flag column - return unchanged."""
        from repogen.reporting.export import _filter_to_headline
        df = self._make_de_df().drop(columns=["passes_headline_min_genes"])
        out = _filter_to_headline(df)
        assert len(out) == 4
        pd.testing.assert_frame_equal(out, df)

    def test_filter_to_headline_handles_none(self) -> None:
        from repogen.reporting.export import _filter_to_headline
        assert _filter_to_headline(None) is None

    def test_filter_to_headline_treats_nan_as_false(self) -> None:
        from repogen.reporting.export import _filter_to_headline
        df = self._make_de_df()
        df["passes_headline_min_genes"] = df["passes_headline_min_genes"].astype(object)
        df.loc[1, "passes_headline_min_genes"] = np.nan
        out = _filter_to_headline(df)
        assert list(out["drug_chembl_id"]) == ["A"]

    def test_export_results_filters_drug_csv(self, tmp_path: Path) -> None:
        from repogen.reporting.export import export_results
        df = self._make_de_df()
        parquet = tmp_path / "x_drug_enrichment.parquet"
        df.to_parquet(parquet, index=False)
        export_results(
            result_type="drug",
            results_path=parquet,
            output_dir=tmp_path,
            study_name="x",
            formats=["csv"],
        )
        out = pd.read_csv(tmp_path / "x_drug.csv")
        assert list(out["drug_chembl_id"]) == ["A", "B"]

    def test_export_results_drug_export_skipped_for_legacy_parquet(
        self, tmp_path: Path,
    ) -> None:
        """Legacy parquet (no flag) is exported unchanged."""
        from repogen.reporting.export import export_results
        df = self._make_de_df().drop(columns=["passes_headline_min_genes"])
        parquet = tmp_path / "y_drug_enrichment.parquet"
        df.to_parquet(parquet, index=False)
        export_results(
            result_type="drug",
            results_path=parquet,
            output_dir=tmp_path,
            study_name="y",
            formats=["csv"],
        )
        out = pd.read_csv(tmp_path / "y_drug.csv")
        assert len(out) == 4

    def test_compute_drug_overlap_filters_subheadline(self) -> None:
        """If a sub-headline drug had a low magma_fdr_q (impossible in practice
        because FDR is NaN, but possible via stale fixtures), the headline
        gate catches it."""
        from repogen.reporting.combine_results import _compute_drug_overlap

        de = self._make_de_df()
        de.loc[2, "magma_fdr_q"] = 0.001
        nc = pd.DataFrame({
            "drug_name": ["X"],
            "drug_chembl_id": ["X"],
            "n_tissues_fdr_significant": [3],
            "best_spearman_rho": [-0.6],
        })
        overlap = _compute_drug_overlap(
            drug_enrichment=de, neg_correlation_summary=nc, mr_drug_matches=None,
        )
        if overlap is None or overlap.empty:
            return
        magma_drugs = overlap.loc[overlap["in_magma"] == True, "drug_chembl_id"].tolist()
        assert "C" not in magma_drugs

    def test_plot_drug_enrichment_filters_subheadline(self) -> None:
        from repogen.plotting.enrichment import plot_drug_enrichment
        df = self._make_de_df()
        df["magma_z"] = [3.0, 2.5, 0.5, -0.2]
        df["magma_beta"] = [0.4, 0.3, 0.05, -0.01]
        meta = {
            "n_drugs_tested": 4,
            "n_drugs_in_atc_pool": 4,
            "n_drugs_in_headline_pool": 2,
        }
        fig = plot_drug_enrichment(df, n_top=20, meta=meta)
        ax = fig.axes[0]
        labels = [t.get_text().strip() for t in ax.get_yticklabels() if t.get_text().strip()]
        assert all(label not in {"C", "D"} for label in labels), labels
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Integration test using real MAGMA binary
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="MAGMA not installed",
)
class TestIntegrationRealMagma:

    def test_real_magma(
        self,
        tmp_path: Path,
        gene_results_df: pd.DataFrame,
        drug_targets_df: pd.DataFrame,
    ) -> None:
        """End-to-end with real MAGMA binary (skipped if not available)."""
        raw_file = tmp_path / "test.genes.raw"
        raw_file.touch()

        config = DrugEnrichmentConfig()
        results = run_drug_enrichment(
            gene_results_raw=raw_file,
            gene_results_df=gene_results_df,
            drug_targets=drug_targets_df,
            config=config,
            output_dir=tmp_path / "output",
            study_name="integration_test",
        )
        assert len(results) > 0
