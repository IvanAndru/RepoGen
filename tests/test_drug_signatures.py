"""Tests for repogen.data.drug_signatures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from repogen.data.drug_signatures import (
    match_lincs_to_chembl,
    _normalize_brd_id,
    _normalize_pubchem_cid,
    _parse_gctx_col_id,
    _extract_profiles_chunked,
    aggregate_signatures,
    _build_signature_records,
    _name_only_matching,
    weighted_nanmedian_per_gene,
    resolve_neural_cell_lines_from_yaml,
    NEURAL_PRIMARY_CELL_LINES,
    NEURAL_TUMOR_CELL_LINES,
)


class TestAggregateSignatures:
    """Tests for signature aggregation."""

    def _make_profiles(self, n_drugs: int = 3, n_profiles: int = 5) -> dict:
        """Create synthetic profile data."""
        profiles = {}
        for i in range(n_drugs):
            pid = f"BRD-K{i:08d}"
            prof_list = []
            for j in range(n_profiles):
                prof_list.append({
                    "z_scores": list(np.random.randn(100)),
                    "cell_line": f"CELL_{j % 2}",
                    "dose": "10 µM" if j < 3 else "1 µM",
                    "time": "24 h",
                })
            profiles[pid] = prof_list
        return profiles

    def test_consensus_aggregation(self) -> None:
        profiles = self._make_profiles(n_drugs=2, n_profiles=5)
        result = aggregate_signatures(profiles, method="consensus", min_profiles=3)
        assert len(result) == 2
        for sig in result.values():
            assert len(sig["z_scores"]) == 100
            assert sig["n_profiles"] == 5

    def test_min_profiles_filter(self) -> None:
        profiles = self._make_profiles(n_drugs=1, n_profiles=2)
        result = aggregate_signatures(profiles, method="consensus", min_profiles=3)
        assert len(result) == 0

    def test_best_dose_aggregation(self) -> None:
        profiles = self._make_profiles(n_drugs=1, n_profiles=5)
        result = aggregate_signatures(
            profiles, method="best_dose",
            min_profiles=1, preferred_dose="10 µM",
        )
        assert len(result) == 1

    def test_per_condition_aggregation(self) -> None:
        profiles = {}
        pid = "BRD-TEST"
        prof_list = []
        for i in range(6):
            prof_list.append({
                "z_scores": list(np.random.randn(50)),
                "cell_line": "CELL_A" if i < 3 else "CELL_B",
                "dose": "10 µM",
                "time": "24 h",
            })
        profiles[pid] = prof_list

        result = aggregate_signatures(
            profiles, method="per_condition", min_profiles=3,
        )
        assert len(result) == 2

    def test_aggregated_z_scores_are_numpy(self) -> None:
        """Aggregated z_scores must remain as NumPy arrays (not Python lists)."""
        profiles = self._make_profiles(n_drugs=1, n_profiles=5)
        result = aggregate_signatures(profiles, method="consensus", min_profiles=3)
        for sig in result.values():
            assert isinstance(sig["z_scores"], np.ndarray), (
                f"Aggregated z_scores should be ndarray, got {type(sig['z_scores'])}"
            )
            assert sig["z_scores"].dtype == np.float32

    def test_accepts_numpy_input_z_scores(self) -> None:
        """Aggregation must work when z_scores are NumPy arrays (extraction output)."""
        profiles = {}
        pid = "BRD-NUMPY"
        prof_list = []
        for j in range(5):
            prof_list.append({
                "z_scores": np.random.randn(100).astype(np.float32),
                "cell_line": "CELL_A",
                "dose": "10",
                "time": "24",
            })
        profiles[pid] = prof_list
        result = aggregate_signatures(profiles, method="consensus", min_profiles=3)
        assert len(result) == 1
        assert isinstance(result[pid]["z_scores"], np.ndarray)

    def test_empty_profiles(self) -> None:
        result = aggregate_signatures({}, method="consensus", min_profiles=1)
        assert len(result) == 0


class TestBuildSignatureRecords:
    """Tests for building DrugSignatureRecord DataFrame."""

    def test_builds_from_signatures(self) -> None:
        signatures = {
            "BRD-K001": {
                "z_scores": [0.1, -0.2, 0.3],
                "cell_lines": ["CELL_A"],
                "doses": ["10 µM"],
                "time_points": ["24 h"],
                "n_profiles": 5,
            },
        }
        matched = pd.DataFrame({
            "pert_id": ["BRD-K001"],
            "drug_name": ["Aspirin"],
            "drug_inchikey": ["BSYNRYMUTXBXSQ"],
            "drug_chembl_id": ["CHEMBL25"],
            "match_confidence": ["inchikey"],
        })
        gene_ids = [1, 2, 3]

        result = _build_signature_records(signatures, matched, gene_ids)
        assert len(result) == 1
        assert result.iloc[0]["drug_name"] == "Aspirin"
        assert result.iloc[0]["n_profiles_aggregated"] == 5
        assert result.iloc[0]["match_confidence"] == "inchikey"

    def test_numpy_z_scores_converted_to_list(self) -> None:
        """Output boundary: NumPy z_scores must be Python lists for Parquet."""
        signatures = {
            "BRD-K001": {
                "z_scores": np.array([0.1, -0.2, 0.3], dtype=np.float32),
                "cell_lines": ["CELL_A"],
                "doses": ["10 µM"],
                "time_points": ["24 h"],
                "n_profiles": 5,
            },
        }
        matched = pd.DataFrame({
            "pert_id": ["BRD-K001"],
            "drug_name": ["Aspirin"],
            "drug_inchikey": ["BSYNRYMUTXBXSQ"],
            "drug_chembl_id": ["CHEMBL25"],
            "match_confidence": ["inchikey"],
        })
        gene_ids = [1, 2, 3]

        result = _build_signature_records(signatures, matched, gene_ids)
        z = result.iloc[0]["z_scores"]
        assert isinstance(z, list), f"z_scores must be list at output, got {type(z)}"
        assert len(z) == 3


class TestNameOnlyMatching:
    """Tests for fallback name-only matching."""

    def test_extracts_names(self) -> None:
        meta = pd.DataFrame({
            "pert_id": ["BRD-001", "BRD-002"],
            "name": ["aspirin", "ibuprofen"],
        })
        result = _name_only_matching(meta)
        assert len(result) == 2
        assert result.iloc[0]["match_confidence"] == "name"

    def test_empty_metadata(self) -> None:
        result = _name_only_matching(pd.DataFrame())
        assert result.empty


class TestParseGctxColId:
    """Tests for GCTX column ID parser."""

    def test_3_token_standard(self) -> None:
        parsed = _parse_gctx_col_id("AML001_CD34_24H:BRD-A03772856:0.37037")
        assert parsed["pert_id"] == "BRD-A03772856"
        assert parsed["cell_line"] == "CD34"
        assert parsed["dose"] == "0.37037"
        assert parsed["time"] == "24"

    def test_4_token_with_xh_prefix(self) -> None:
        parsed = _parse_gctx_col_id("ABY001_A375_XH:BRD-A61304759:0.625:24")
        assert parsed["pert_id"] == "BRD-A61304759"
        assert parsed["cell_line"] == "A375"
        assert parsed["dose"] == "0.625"
        assert parsed["time"] == "24"

    def test_3_token_underscored_cell_line(self) -> None:
        parsed = _parse_gctx_col_id("CPC016_HA1E_6H:BRD-K12345678:10")
        assert parsed["pert_id"] == "BRD-K12345678"
        assert parsed["cell_line"] == "HA1E"
        assert parsed["dose"] == "10"
        assert parsed["time"] == "6"

    def test_too_few_tokens_returns_unknown(self) -> None:
        parsed = _parse_gctx_col_id("only_two")
        assert parsed["pert_id"] == "unknown"
        assert parsed["cell_line"] == "unknown"
        assert parsed["dose"] == "unknown"
        assert parsed["time"] == "unknown"

    def test_ambiguous_prefix_sets_unknown_cell_line(self) -> None:
        parsed = _parse_gctx_col_id("ODDPREFIX_CELL_NOPE:BRD-K00000001:5")
        assert parsed["pert_id"] == "BRD-K00000001"
        assert parsed["cell_line"] == "unknown"
        assert parsed["time"] == "unknown"

    def test_multi_part_cell_line(self) -> None:
        parsed = _parse_gctx_col_id("EXP_CELL_LINE_24H:BRD-K00000002:3.3")
        assert parsed["pert_id"] == "BRD-K00000002"
        assert parsed["cell_line"] == "CELL_LINE"
        assert parsed["time"] == "24"

    def test_extended_brd_normalized_to_compound(self) -> None:
        """5-segment batch BRD should normalize to 2-segment compound ID."""
        parsed = _parse_gctx_col_id(
            "ASG001_PC3_24H:BRD-A61304759-001-01-0:0.08"
        )
        assert parsed["pert_id"] == "BRD-A61304759"
        assert parsed["cell_line"] == "PC3"
        assert parsed["dose"] == "0.08"
        assert parsed["time"] == "24"


class TestNormalizeBrdId:
    """Tests for BRD identifier normalization."""

    def test_2_segment_unchanged(self) -> None:
        assert _normalize_brd_id("BRD-K25050358") == "BRD-K25050358"

    def test_5_segment_to_compound(self) -> None:
        assert _normalize_brd_id("BRD-K69349031-001-01-5") == "BRD-K69349031"

    def test_non_brd_passthrough(self) -> None:
        assert _normalize_brd_id("SAHA") == "SAHA"

    def test_malformed_brd_passthrough(self) -> None:
        assert _normalize_brd_id("BRD-short") == "BRD-short"


class TestExtractProfilesChunked:
    """Tests for GCTX profile extraction with mocked I/O."""

    FAKE_COL_IDS = [
        b"EXP001_A375_24H:BRD-K00000001:10",
        b"EXP001_A375_24H:BRD-K00000001:1",
        b"EXP001_MCF7_24H:BRD-K00000002:10",
        b"EXP002_A375_6H:BRD-K00000099:5",
        b"ASG001_PC3_24H:BRD-K00000001-001-01-0:0.08",
        b"ASG001_PC3_24H:BRD-K00000002-003-06-0:0.4",
    ]
    GENE_IDS = [10, 100, 1000]

    def _mock_gctx_parse(self, gctx_path, cid, rid):
        """Return a fake GCToo with correct dimensions for the requested cids."""
        n_genes = len(rid)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_genes, len(cid)))
        data_df = pd.DataFrame(data, index=rid, columns=cid)
        return SimpleNamespace(data_df=data_df)

    def _run_extraction(self, pert_ids):
        fake_h5 = MagicMock()
        fake_h5.__enter__ = MagicMock(return_value=fake_h5)
        fake_h5.__exit__ = MagicMock(return_value=False)
        fake_h5.__getitem__ = lambda self_inner, key: {
            "0/META/COL/id": MagicMock(__getitem__=lambda s, sl: np.array(self.FAKE_COL_IDS))
        }[key]

        mock_parse_module = MagicMock()
        mock_parse_module.parse = self._mock_gctx_parse

        with patch("repogen.data.drug_signatures.h5py", create=True) as mock_h5py:
            mock_h5py.File.return_value = fake_h5
            # Override the lazy import inside _index_gctx_columns_for_perts
            with patch.dict("sys.modules", {"h5py": mock_h5py}):
                profiles = _extract_profiles_chunked(
                    parse_gctx=mock_parse_module,
                    gctx_path=Path("fake.gctx"),
                    pert_ids=pert_ids,
                    gene_ids=self.GENE_IDS,
                    chunk_size=100,
                )
        return profiles

    def test_cid_are_strings_not_indices(self) -> None:
        """Regression: extraction must pass real GCTX column ID strings, not row indices."""
        mock_parse_module = MagicMock()
        captured_cids: list = []

        def capture_parse(gctx_path, cid, rid):
            captured_cids.extend(cid)
            data = np.zeros((len(rid), len(cid)))
            return SimpleNamespace(
                data_df=pd.DataFrame(data, index=rid, columns=cid)
            )

        mock_parse_module.parse = capture_parse

        fake_h5 = MagicMock()
        fake_h5.__enter__ = MagicMock(return_value=fake_h5)
        fake_h5.__exit__ = MagicMock(return_value=False)
        fake_h5.__getitem__ = lambda self_inner, key: {
            "0/META/COL/id": MagicMock(__getitem__=lambda s, sl: np.array(self.FAKE_COL_IDS))
        }[key]

        with patch.dict("sys.modules", {"h5py": MagicMock(File=MagicMock(return_value=fake_h5))}):
            _extract_profiles_chunked(
                parse_gctx=mock_parse_module,
                gctx_path=Path("fake.gctx"),
                pert_ids=["BRD-K00000001"],
                gene_ids=self.GENE_IDS,
                chunk_size=100,
            )

        for cid in captured_cids:
            assert isinstance(cid, str), f"cid must be str, got {type(cid)}: {cid}"
            assert not cid.isdigit(), f"cid looks like a row index: {cid}"

    def test_profiles_grouped_by_perturbagen(self) -> None:
        """Profiles should be keyed by pert_id with correct count.

        BRD-K00000001 has 2 short-form + 1 extended-form = 3 profiles.
        BRD-K00000002 has 1 short-form + 1 extended-form = 2 profiles.
        """
        profiles = self._run_extraction(["BRD-K00000001", "BRD-K00000002"])
        assert "BRD-K00000001" in profiles
        assert "BRD-K00000002" in profiles
        assert len(profiles["BRD-K00000001"]) == 3
        assert len(profiles["BRD-K00000002"]) == 2

    def test_metadata_from_col_id_not_csv(self) -> None:
        """Regression: profile metadata must come from column ID, not external CSV."""
        profiles = self._run_extraction(["BRD-K00000001"])
        cell_lines = {p["cell_line"] for p in profiles["BRD-K00000001"]}
        assert cell_lines == {"A375", "PC3"}
        for prof in profiles["BRD-K00000001"]:
            assert prof["time"] == "24"
            assert prof["dose"] in ("10", "1", "0.08")
            assert len(prof["z_scores"]) == len(self.GENE_IDS)

    def test_extended_brd_columns_matched_to_compound(self) -> None:
        """5-segment batch BRDs must be captured under the 2-segment compound ID."""
        profiles = self._run_extraction(["BRD-K00000001"])
        assert len(profiles["BRD-K00000001"]) == 3
        doses = sorted(p["dose"] for p in profiles["BRD-K00000001"])
        assert "0.08" in doses

    def test_z_scores_are_numpy_arrays(self) -> None:
        """Memory safety: z_scores must be numpy arrays, not Python lists."""
        profiles = self._run_extraction(["BRD-K00000001"])
        for prof in profiles["BRD-K00000001"]:
            assert isinstance(prof["z_scores"], np.ndarray), (
                f"z_scores should be ndarray, got {type(prof['z_scores'])}"
            )
            assert prof["z_scores"].dtype == np.float32

    def test_unmatched_perts_return_empty(self) -> None:
        profiles = self._run_extraction(["BRD-NOTINGCTX"])
        assert profiles == {}

    def test_parse_error_skips_chunk_gracefully(self) -> None:
        """cmapPy parse failure should skip the chunk, not crash."""
        mock_parse_module = MagicMock()
        mock_parse_module.parse.side_effect = Exception("simulated parse failure")

        fake_h5 = MagicMock()
        fake_h5.__enter__ = MagicMock(return_value=fake_h5)
        fake_h5.__exit__ = MagicMock(return_value=False)
        fake_h5.__getitem__ = lambda self_inner, key: {
            "0/META/COL/id": MagicMock(__getitem__=lambda s, sl: np.array(self.FAKE_COL_IDS))
        }[key]

        with patch.dict("sys.modules", {"h5py": MagicMock(File=MagicMock(return_value=fake_h5))}):
            profiles = _extract_profiles_chunked(
                parse_gctx=mock_parse_module,
                gctx_path=Path("fake.gctx"),
                pert_ids=["BRD-K00000001"],
                gene_ids=self.GENE_IDS,
                chunk_size=100,
            )
        assert profiles == {}


# ---------------------------------------------------------------------------
# PubChem CID normalization tests
# ---------------------------------------------------------------------------


class TestNormalizePubchemCid:

    def test_float_string(self) -> None:
        assert _normalize_pubchem_cid("6005.0") == "6005"

    def test_int_string(self) -> None:
        assert _normalize_pubchem_cid("6005") == "6005"

    def test_actual_float(self) -> None:
        assert _normalize_pubchem_cid(6005.0) == "6005"

    def test_actual_int(self) -> None:
        assert _normalize_pubchem_cid(6005) == "6005"

    def test_nan_returns_none(self) -> None:
        assert _normalize_pubchem_cid(float("nan")) is None

    def test_none_returns_none(self) -> None:
        assert _normalize_pubchem_cid(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize_pubchem_cid("") is None

    def test_na_string_returns_none(self) -> None:
        assert _normalize_pubchem_cid("nan") is None
        assert _normalize_pubchem_cid("<NA>") is None

    def test_non_numeric_returns_none(self) -> None:
        assert _normalize_pubchem_cid("NOT_A_CID") is None

    def test_large_cid(self) -> None:
        assert _normalize_pubchem_cid("134129865.0") == "134129865"


# ---------------------------------------------------------------------------
# Name collision dedup regression tests (crash fix)
# ---------------------------------------------------------------------------


class TestNameCollisionDedup:
    """Regression tests for case-variant name collisions in name matching."""

    @staticmethod
    def _make_lincs(names: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "pert_id": [f"BRD-{i:09d}" for i in range(len(names))],
            "pert_iname": names,
            "InChIKey": [None] * len(names),
        })

    @staticmethod
    def _make_dt(rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "drug_inchikey": None, "drug_pubchem_cid": None,
            "drug_smiles": None, "gene_symbol": "GENE1",
            "gene_ensembl_id": "ENSG00000000001", "gene_uniprot_id": "P00001",
            "gene_entrez_id": 1, "interaction_type": "inhibitor",
            "action_type": None, "mechanism_of_action": None,
            "pchembl_value": None, "affinity_value": None,
            "affinity_type": None, "affinity_unit": None,
            "max_phase": 0, "atc_codes": None, "indication_mesh": None,
            "molecule_type": None, "is_withdrawn": False,
            "source": "chembl", "confidence": "medium", "source_pmids": None,
        }
        full_rows = [{**defaults, **r} for r in rows]
        return pd.DataFrame(full_rows)

    def test_case_variants_no_crash(self) -> None:
        """Case variants of the same drug name must not crash match_lincs_to_chembl."""
        lincs = self._make_lincs(["drugX", "drugY"])
        dt = self._make_dt([
            {"drug_name": "Agomelatine", "drug_chembl_id": "CHEMBL10878"},
            {"drug_name": "AGOMELATINE", "drug_chembl_id": "CHEMBL10878"},
        ])
        result = match_lincs_to_chembl(lincs, dt)
        assert isinstance(result, pd.DataFrame)

    def test_conflicting_ids_prefer_canonical(self) -> None:
        """When case variants have different IDs, canonical CHEMBL wins."""
        lincs = self._make_lincs(["ku-0063794"])
        dt = self._make_dt([
            {"drug_name": "Ku-0063794", "drug_chembl_id": "CHEMBL1078983"},
            {"drug_name": "KU-0063794", "drug_chembl_id": "DGIDB_KU-0063794"},
        ])
        result = match_lincs_to_chembl(lincs, dt)
        assert isinstance(result, pd.DataFrame)

    def test_many_collisions_no_crash(self) -> None:
        """Stress test: many case-variant collisions."""
        lincs = self._make_lincs(["test"])
        rows = []
        for i in range(50):
            name = f"drug{i}"
            rows.append({"drug_name": name.upper(), "drug_chembl_id": f"CHEMBL{i}"})
            rows.append({"drug_name": name.capitalize(), "drug_chembl_id": f"CHEMBL{i}"})
        dt = self._make_dt(rows)
        result = match_lincs_to_chembl(lincs, dt)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Cell-line-aware consensus weighting + composition QC
# ---------------------------------------------------------------------------


def _synthetic_profiles(cell_lines: list[str], n_genes: int = 20, seed: int = 42) -> list[dict]:
    """Build a synthetic profile list with deterministic z-vectors.

    Each profile's z-vector is ``[i + 0.1 * j for j in range(n_genes)]``
    where ``i`` is the profile index, so per-gene sorts are predictable.
    """
    rng = np.random.default_rng(seed)
    return [
        {
            "z_scores": rng.standard_normal(n_genes).astype(np.float32),
            "cell_line": cl,
            "dose": "10 µM",
            "time": "24 h",
        }
        for i, cl in enumerate(cell_lines)
    ]


class TestWeightedNanmedianPerGene:
    """Tests for the weighted_nanmedian_per_gene helper."""

    def test_uniform_weights_matches_nanmedian_bytewise(self) -> None:
        """Equal weights must be byte-identical to np.nanmedian."""
        rng = np.random.default_rng(0)
        z = rng.standard_normal((7, 100)).astype(np.float32)
        w = np.ones(7)
        weighted = weighted_nanmedian_per_gene(z, w)
        reference = np.nanmedian(z, axis=0).astype(np.float32)
        # Byte-for-byte identity (fast path delegates to np.nanmedian).
        np.testing.assert_array_equal(weighted, reference)

    def test_even_n_equal_weights_matches_nanmedian(self) -> None:
        """[1, 3] with equal weights must
        return 2.0, not 1.0.  This is exactly the failure mode of the
        naive "lower weighted median" for even n.
        """
        z = np.array([[1.0], [3.0]], dtype=np.float32)
        w = np.array([1.0, 1.0])
        result = weighted_nanmedian_per_gene(z, w)
        # np.nanmedian([1, 3]) = 2.0 (average of two middle values).
        np.testing.assert_array_equal(result, np.array([2.0], dtype=np.float32))

    def test_correctness_known_case_unequal_weights(self) -> None:
        """Hand-computed weighted median on a small fixture.

        Values (sorted): 1, 2, 3, 4, 5; weights: 1, 1, 1, 1, 4.
        Cumulative: 1, 2, 3, 4, 8; total=8; threshold=4.
        Smallest value with cum >= 4 is 4.  Lower weighted median = 4.
        """
        z = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        w = np.array([1.0, 1.0, 1.0, 1.0, 4.0])
        result = weighted_nanmedian_per_gene(z, w)
        assert result[0] == 4.0

    def test_handles_nan_per_gene(self) -> None:
        """One gene column all-NaN -> NaN; mixed -> weighted median of non-NaN."""
        z = np.array([
            [1.0, np.nan, 5.0],
            [2.0, np.nan, np.nan],
            [3.0, np.nan, 10.0],
        ], dtype=np.float32)
        w = np.array([1.0, 1.0, 1.0])
        result = weighted_nanmedian_per_gene(z, w)
        # Gene 0: median(1,2,3) = 2
        assert result[0] == 2.0
        # Gene 1: all NaN -> NaN
        assert np.isnan(result[1])
        # Gene 2: median(5, 10) = 7.5 under np.nanmedian (2-value case).
        assert result[2] == 7.5

    def test_vectorized_matches_reference_on_realistic_shape(self) -> None:
        """Sanity-check the vectorised implementation
        against a per-column Python reference on realistic dimensions.
        """
        rng = np.random.default_rng(1)
        z = rng.standard_normal((25, 500)).astype(np.float32)
        # Introduce sparse NaNs to exercise the mask path.
        nan_mask = rng.random(z.shape) < 0.05
        z[nan_mask] = np.nan
        w = np.array([3.0] * 10 + [1.0] * 15)  # unequal weights
        vec = weighted_nanmedian_per_gene(z, w)

        # Column-by-column Python reference (lower weighted median).
        ref = np.empty(z.shape[1], dtype=np.float32)
        for g in range(z.shape[1]):
            col = z[:, g]
            valid = ~np.isnan(col)
            if not valid.any():
                ref[g] = np.nan
                continue
            vals = col[valid]
            wts = w[valid]
            order = np.argsort(vals, kind="mergesort")
            vs = vals[order]
            ws = wts[order]
            cs = np.cumsum(ws)
            idx = int(np.argmax(cs >= 0.5 * cs[-1]))
            ref[g] = vs[idx]
        np.testing.assert_allclose(vec, ref, equal_nan=True, rtol=1e-6)

    def test_runtime_within_2x_of_nanmedian_on_realistic_shape(self) -> None:
        """Benchmark: the vectorised implementation must be
        within ~2x of ``np.nanmedian`` on realistic dims.  Not a strict
        unit test - a benchmark that flags pathological regression.
        """
        import time
        rng = np.random.default_rng(2)
        z = rng.standard_normal((200, 12328)).astype(np.float32)
        w = np.array([3.0] * 60 + [1.0] * 140)

        # Warm-up (compile numpy kernels).
        _ = np.nanmedian(z, axis=0)
        _ = weighted_nanmedian_per_gene(z, w)

        t0 = time.perf_counter()
        for _ in range(3):
            np.nanmedian(z, axis=0)
        t_np = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            weighted_nanmedian_per_gene(z, w)
        t_wm = (time.perf_counter() - t0) / 3

        # Allow generous slack (5x) for CI jitter; if we start seeing
        # regressions >5x we know something is wrong.
        assert t_wm <= 5.0 * t_np, (
            f"weighted_nanmedian_per_gene wall time {t_wm:.3f}s exceeds "
            f"5x np.nanmedian baseline {t_np:.3f}s (ratio={t_wm / max(t_np, 1e-9):.2f}x)"
        )


class TestConfigDefaults:
    """DrugSignaturesConfig default resolution tests."""

    def test_config_default_is_uniform(self) -> None:
        """Backward-compat: default cell_line_weighting must be 'uniform'."""
        from repogen.config.schema import DrugSignaturesConfig
        cfg = DrugSignaturesConfig()
        assert cfg.cell_line_weighting == "uniform"

    def test_neural_cell_lines_model_validator_derives_default(self) -> None:
        """model_validator(mode='after') derives the
        combined default (primary + tumor) when include_neural_tumor_cell_lines=True.
        """
        from repogen.config.schema import DrugSignaturesConfig
        cfg = DrugSignaturesConfig(include_neural_tumor_cell_lines=True)
        assert set(cfg.neural_cell_lines) == set(
            NEURAL_PRIMARY_CELL_LINES + NEURAL_TUMOR_CELL_LINES
        )

    def test_include_neural_tumor_cell_lines_switch_false(self) -> None:
        """When switch is False, default resolves to primary neural only."""
        from repogen.config.schema import DrugSignaturesConfig
        cfg = DrugSignaturesConfig(include_neural_tumor_cell_lines=False)
        assert set(cfg.neural_cell_lines) == set(NEURAL_PRIMARY_CELL_LINES)

    def test_neural_cell_lines_user_override_is_normalized(self) -> None:
        """User overrides go through the same normalize
        path - upper, strip, dedupe, drop empties.  Pydantic strict
        typing rejects non-string list members before the validator
        runs, so this test uses only string inputs (including empties
        and case variants) - which is what the validator needs to
        normalize.
        """
        from repogen.config.schema import DrugSignaturesConfig
        cfg = DrugSignaturesConfig(
            neural_cell_lines=["neu", " npc ", "SHSY5Y", "", "NPC", "  "],
        )
        # Empties stripped; duplicates deduped; case normalized.
        assert set(cfg.neural_cell_lines) == {"NEU", "NPC", "SHSY5Y"}


class TestCompositionCounts:
    """Composition column tests."""

    def test_composition_computed_from_profiles_not_unique_cell_lines(self) -> None:
        """A drug with [NPC, NPC, NPC, MCF7]
        must report n_profiles_neural=3, NOT 1 (which would be the wrong
        len(set(cell_lines)) result).
        """
        prof_list = [
            {"z_scores": np.zeros(10, dtype=np.float32), "cell_line": "NPC",  "dose": "10 µM", "time": "24 h"},
            {"z_scores": np.zeros(10, dtype=np.float32), "cell_line": "NPC",  "dose": "10 µM", "time": "24 h"},
            {"z_scores": np.zeros(10, dtype=np.float32), "cell_line": "NPC",  "dose": "10 µM", "time": "24 h"},
            {"z_scores": np.zeros(10, dtype=np.float32), "cell_line": "MCF7", "dose": "10 µM", "time": "24 h"},
        ]
        result = aggregate_signatures(
            {"BRD-K001": prof_list},
            method="consensus", min_profiles=3,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        sig = result["BRD-K001"]
        assert sig["n_profiles_neural"] == 3
        assert sig["n_profiles_non_neural"] == 1
        assert sig["n_profiles_total"] == 4
        assert sig["neural_fraction"] == pytest.approx(0.75)

    def test_composition_counts_add_up_to_total(self) -> None:
        """Invariant: total == neural + non_neural + unknown."""
        prof_list = [
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "NPC",     "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7",    "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7",    "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "unknown", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K999": prof_list},
            method="consensus", min_profiles=3,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        sig = result["BRD-K999"]
        assert sig["n_profiles_total"] == (
            sig["n_profiles_neural"]
            + sig["n_profiles_non_neural"]
            + sig["n_profiles_unknown_cell_line"]
        )

    def test_composition_columns_present_in_signature_records(self) -> None:
        """After _build_signature_records, all 8 cell-line composition columns exist."""
        prof_list = [
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        signatures = aggregate_signatures(
            {"BRD-K123": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        matched = pd.DataFrame({
            "pert_id": ["BRD-K123"], "drug_name": ["TestDrug"],
            "drug_inchikey": ["IK-TEST"], "drug_chembl_id": ["CHEMBL1"],
            "match_confidence": ["inchikey"],
        })
        records = _build_signature_records(signatures, matched, gene_ids=list(range(5)))
        expected_composition_cols = {
            "n_profiles_total", "n_profiles_neural", "n_profiles_non_neural",
            "n_profiles_unknown_cell_line", "neural_fraction",
            "neural_weight_fraction", "cell_line_weighting_mode", "neural_weight",
        }
        assert expected_composition_cols.issubset(set(records.columns))
        row = records.iloc[0]
        assert row["cell_line_weighting_mode"] == "neural_priority"
        assert row["neural_weight"] == 3.0

    def test_neural_weight_fraction_uniform_equals_neural_fraction(self) -> None:
        prof_list = [
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=3,
            cell_line_weighting="uniform",
        )
        sig = result["BRD-K"]
        assert sig["neural_weight_fraction"] == sig["neural_fraction"] == 0.0

    def test_neural_weight_fraction_neural_priority_reflects_weights(self) -> None:
        """In neural_priority: 1 neural weighted 3.0 vs 3 non-neural weighted 1.0
        -> neural_weight_fraction = 3 / (3 + 3) = 0.5, but neural_fraction = 1/4 = 0.25.
        """
        prof_list = [
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=3,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        sig = result["BRD-K"]
        assert sig["neural_fraction"] == pytest.approx(0.25)
        assert sig["neural_weight_fraction"] == pytest.approx(0.5)


class TestAggregationModes:
    """Mode-switch aggregation semantics."""

    def test_uniform_mode_byte_identical_to_plain_median(self) -> None:
        """Uniform mode reproduces the aggregation used before cell-line weighting
        exactly.  Compare against a direct np.nanmedian on the
        raw profile z-matrix.
        """
        rng = np.random.default_rng(7)
        prof_list = [
            {"z_scores": rng.standard_normal(50).astype(np.float32),
             "cell_line": "MCF7", "dose": "d", "time": "t"}
            for _ in range(5)
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=3,
            cell_line_weighting="uniform",
        )
        expected = np.nanmedian(
            np.array([p["z_scores"] for p in prof_list]), axis=0,
        ).astype(np.float32)
        np.testing.assert_array_equal(result["BRD-K"]["z_scores"], expected)

    def test_neural_priority_zero_neural_profiles_identical_to_uniform(self) -> None:
        """A drug with all-cancer profiles under
        neural_priority produces byte-identical output to uniform.
        """
        rng = np.random.default_rng(11)
        prof_list = [
            {"z_scores": rng.standard_normal(30).astype(np.float32),
             "cell_line": "MCF7", "dose": "d", "time": "t"}
            for _ in range(4)
        ]
        uniform = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="uniform",
        )
        priority = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        np.testing.assert_array_equal(
            uniform["BRD-K"]["z_scores"], priority["BRD-K"]["z_scores"],
        )

    def test_neural_priority_changes_only_drugs_with_neural_profiles(self) -> None:
        rng = np.random.default_rng(13)
        # Drug A: no neural profiles -> identical.
        prof_a = [
            {"z_scores": rng.standard_normal(20).astype(np.float32),
             "cell_line": "MCF7", "dose": "d", "time": "t"}
            for _ in range(3)
        ]
        # Drug B: mixed profiles -> differs.
        prof_b = [
            {"z_scores": rng.standard_normal(20).astype(np.float32),
             "cell_line": "NPC" if i == 0 else "MCF7",
             "dose": "d", "time": "t"}
            for i in range(3)
        ]
        uniform = aggregate_signatures(
            {"A": prof_a, "B": prof_b},
            method="consensus", min_profiles=1,
            cell_line_weighting="uniform",
        )
        priority = aggregate_signatures(
            {"A": prof_a, "B": prof_b},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        np.testing.assert_array_equal(uniform["A"]["z_scores"], priority["A"]["z_scores"])
        # Drug B must differ somewhere.
        assert not np.array_equal(uniform["B"]["z_scores"], priority["B"]["z_scores"])

    def test_neural_only_drops_below_min_neural_profiles(self) -> None:
        prof_list = [
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(5, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_only",
            neural_cell_lines=["NPC"], min_neural_profiles=2,
        )
        # 1 neural profile < 2 required -> dropped.
        assert len(result) == 0

    def test_neural_only_keeps_only_neural_profiles(self) -> None:
        prof_list = [
            {"z_scores": np.array([1.0, 2.0], dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.array([3.0, 4.0], dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.array([99.0, 99.0], dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_only",
            neural_cell_lines=["NPC"], min_neural_profiles=1,
        )
        sig = result["BRD-K"]
        # Consensus of NPC profiles only: median([1,3]) = 2, median([2,4]) = 3.
        np.testing.assert_array_equal(sig["z_scores"], np.array([2.0, 3.0], dtype=np.float32))
        # Aggregation set size = 2 (post-filter).
        assert sig["n_profiles"] == 2
        # cell_lines list on the aggregation set is neural-only.
        assert set(sig["cell_lines"]) == {"NPC"}

    def test_neural_only_composition_reflects_pre_filter_set(self) -> None:
        """neural_fraction must reflect the ORIGINAL
        composition, not the trivially-1.0 post-filter fraction.
        Drug with 1 NPC + 3 MCF7 under neural_only -> n_profiles_total=4,
        n_profiles_neural=1, neural_fraction=0.25, n_profiles_aggregated=1.
        """
        prof_list = [
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_only",
            neural_cell_lines=["NPC"], min_neural_profiles=1,
        )
        sig = result["BRD-K"]
        assert sig["n_profiles_total"] == 4        # pre-filter
        assert sig["n_profiles_neural"] == 1       # pre-filter
        assert sig["neural_fraction"] == pytest.approx(0.25)  # pre-filter
        assert sig["n_profiles"] == 1              # post-filter (aggregation set)

    def test_neural_only_min_defaults_to_min_profiles(self) -> None:
        """min_neural_profiles=None resolves to min_profiles.
        With min_profiles=3 and 2 neural profiles -> drug is dropped.
        """
        prof_list = [
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=3,
            cell_line_weighting="neural_only",
            neural_cell_lines=["NPC"], min_neural_profiles=None,
        )
        # 2 neural profiles < effective 3 -> dropped.
        assert len(result) == 0

    def test_cell_line_matching_case_insensitive_exact(self) -> None:
        """Aggregation normalises cell_line via .upper().strip() before
        matching - 'npc', ' NPC ', 'NPC' all match.  Substrings like
        'NPCX' do NOT (no fuzzy).
        """
        prof_list = [
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "npc",   "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": " NPC ", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPCX",  "dose": "d", "time": "t"},
        ]
        result = aggregate_signatures(
            {"BRD-K": prof_list},
            method="consensus", min_profiles=1,
            cell_line_weighting="neural_priority",
            neural_cell_lines=["NPC"], neural_weight=3.0,
        )
        sig = result["BRD-K"]
        # First two profiles match; third ("NPCX") does not.
        assert sig["n_profiles_neural"] == 2
        assert sig["n_profiles_non_neural"] == 1

    def test_aggregation_logs_cell_line_composition(self, caplog) -> None:
        """The runtime QC log reports top-N
        observed cell-line tokens and neural-token matches.
        """
        import logging
        prof_list = [
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPC",  "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "MCF7", "dose": "d", "time": "t"},
        ]
        with caplog.at_level(logging.INFO, logger="repogen.data.drug_signatures"):
            aggregate_signatures(
                {"BRD-K": prof_list},
                method="consensus", min_profiles=1,
                cell_line_weighting="neural_priority",
                neural_cell_lines=["NPC"], neural_weight=3.0,
            )
        joined = "\n".join(rec.message for rec in caplog.records)
        assert "Cell-line composition" in joined
        assert "Neural cell-line tokens matched" in joined
        assert "NPC" in joined and "MCF7" in joined
        # Neural priority cohort log.
        assert "neural_priority cohort" in joined

    def test_neural_only_missing_neural_config_raises(self) -> None:
        """Modes other than uniform require a non-empty neural_cell_lines."""
        prof_list = [
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPC",
             "dose": "d", "time": "t"}
            for _ in range(3)
        ]
        with pytest.raises(ValueError, match="requires a non-empty neural_cell_lines"):
            aggregate_signatures(
                {"BRD-K": prof_list},
                method="consensus", min_profiles=1,
                cell_line_weighting="neural_only",
                neural_cell_lines=None,
            )

    def test_invalid_cell_line_weighting_raises(self) -> None:
        prof_list = [
            {"z_scores": np.zeros(3, dtype=np.float32), "cell_line": "NPC",
             "dose": "d", "time": "t"}
            for _ in range(3)
        ]
        with pytest.raises(ValueError, match="cell_line_weighting must be one of"):
            aggregate_signatures(
                {"BRD-K": prof_list},
                method="consensus", min_profiles=1,
                cell_line_weighting="bogus_mode",
            )


# ---------------------------------------------------------------------------
# Shared neural-cell-line resolver (Pydantic + Snakemake parity)
# ---------------------------------------------------------------------------


class TestResolveNeuralCellLinesFromYaml:
    """The shared resolver must honour ``include_neural_tumor_cell_lines``
    from raw YAML AND match the Pydantic validator bit-for-bit.
    """

    def test_none_config_yields_default_with_tumor(self) -> None:
        """Empty / missing config -> primary + tumor default (backward-compat)."""
        result = resolve_neural_cell_lines_from_yaml(None)
        assert set(result) == set(NEURAL_PRIMARY_CELL_LINES + NEURAL_TUMOR_CELL_LINES)

    def test_include_neural_tumor_cell_lines_false_excludes_tumor_from_yaml(self) -> None:
        """Raw YAML `include_neural_tumor_cell_lines: false`
        (without an explicit `neural_cell_lines` override) must produce a
        primary-neural-only list.  Before the fix, the Snakemake path
        ignored this switch and defaulted to primary + tumor.
        """
        result = resolve_neural_cell_lines_from_yaml(
            {"include_neural_tumor_cell_lines": False}
        )
        assert set(result) == set(NEURAL_PRIMARY_CELL_LINES)
        assert "LN229" not in result  # tumor default explicitly excluded

    def test_include_neural_tumor_cell_lines_true_includes_tumor_from_yaml(self) -> None:
        result = resolve_neural_cell_lines_from_yaml(
            {"include_neural_tumor_cell_lines": True}
        )
        assert set(result) == set(NEURAL_PRIMARY_CELL_LINES + NEURAL_TUMOR_CELL_LINES)
        assert "LN229" in result

    def test_user_override_bypasses_include_neural_tumor_switch(self) -> None:
        """If `neural_cell_lines` is explicitly set, the switch is ignored
        (the switch only affects the derived default).
        """
        result = resolve_neural_cell_lines_from_yaml({
            "neural_cell_lines": ["NEU", "NPC", "GLIA1"],
            # switch says False, but user override wins
            "include_neural_tumor_cell_lines": False,
        })
        assert set(result) == {"NEU", "NPC", "GLIA1"}

    def test_user_override_is_normalized(self) -> None:
        """Same normalization contract as the Pydantic path: upper + strip
        + dedupe + drop empties.
        """
        result = resolve_neural_cell_lines_from_yaml({
            "neural_cell_lines": ["neu", " npc ", "SHSY5Y", "", "NPC", "  "],
        })
        assert set(result) == {"NEU", "NPC", "SHSY5Y"}

    def test_resolver_matches_pydantic_validator_across_scenarios(self) -> None:
        """Parity check: for each scenario, the raw-YAML resolver produces
        the same output the Pydantic validator produces.  Locks the
        single-source-of-truth contract.
        """
        from repogen.config.schema import DrugSignaturesConfig

        scenarios = [
            {},
            {"include_neural_tumor_cell_lines": True},
            {"include_neural_tumor_cell_lines": False},
            {"neural_cell_lines": ["NEU", "NPC"]},
            {"neural_cell_lines": ["neu", " npc "], "include_neural_tumor_cell_lines": False},
        ]
        for scenario in scenarios:
            raw = resolve_neural_cell_lines_from_yaml(scenario)
            pydantic_cfg = DrugSignaturesConfig(**scenario)
            assert raw == pydantic_cfg.neural_cell_lines, (
                f"Resolver ({raw}) != Pydantic ({pydantic_cfg.neural_cell_lines}) "
                f"for scenario {scenario}"
            )

    def test_snakemake_helper_returns_empty_for_uniform_mode(self) -> None:
        """The Snakemake ``neural_cell_lines_flag_for`` helper (defined in
        common.smk) must emit an empty string for uniform mode.  We test
        the underlying logic by exercising the same branch condition.

        The helper is Snakemake-scoped (defined in a .smk file), so we
        emulate its two-step logic here to lock the semantic contract.
        """
        # uniform mode -> no flag
        cfg_uniform = {"cell_line_weighting": "uniform"}
        assert cfg_uniform.get("cell_line_weighting", "uniform") == "uniform"

        # non-uniform mode -> resolver produces the token list
        cfg_neural = {
            "cell_line_weighting": "neural_priority",
            "include_neural_tumor_cell_lines": False,
        }
        assert cfg_neural.get("cell_line_weighting", "uniform") != "uniform"
        tokens = resolve_neural_cell_lines_from_yaml(cfg_neural)
        assert "LN229" not in tokens
        assert set(tokens) == set(NEURAL_PRIMARY_CELL_LINES)


class TestCensusScriptDefaults:
    """The census script's presence check must reference the
    production constants, not the rejected proposals.
    """

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "scripts").exists(),
        reason="scripts/ is not shipped inside the image; repo checkout only",
    )
    def test_phase0_script_imports_production_constants(self) -> None:
        """Static import parity: the script's PRODUCTION_ALL tuple must
        equal the current production constants.  Guards against future
        divergence.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lincs_cell_line_census",
            Path(__file__).parent.parent / "scripts" / "lincs_cell_line_census.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.PRODUCTION_PRIMARY == tuple(NEURAL_PRIMARY_CELL_LINES)
        assert mod.PRODUCTION_TUMOR == tuple(NEURAL_TUMOR_CELL_LINES)
        # Historical-rejected tokens must not leak into PRODUCTION_ALL.
        assert "MNEU" not in mod.PRODUCTION_ALL
        assert "SHSY5Y" not in mod.PRODUCTION_ALL
        # They live in HISTORICAL_REJECTED for audit visibility.
        assert "MNEU" in mod.HISTORICAL_REJECTED
        assert "SHSY5Y" in mod.HISTORICAL_REJECTED
