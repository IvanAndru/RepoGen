"""Branch C's drug layer with route-labelled drug tables (records 041 to 048)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repogen.analysis.mendelian_randomisation import (
    _summarise_gene_verdict,
    drug_layer_calibration,
    match_drugs_for_eligible,
    match_drugs_to_mr_gene,
    summarise_drug_compounds,
)
from repogen.config.schema import MRDrugMatchConfig
from repogen.data.chembl_drug_info import ChemblDrugInfo, dgidb_direction
from repogen.reporting.combine_results import primary_mr_matches

_ACTION_CLASSES = {
    "INHIBITOR": "NEGATIVE MODULATOR", "ANTAGONIST": "NEGATIVE MODULATOR",
    "AGONIST": "POSITIVE MODULATOR", "PARTIAL AGONIST": "POSITIVE MODULATOR",
    "BLOCKER": "NEGATIVE MODULATOR", "MODULATOR": "OTHER",
}


def _info(**kw) -> ChemblDrugInfo:
    return ChemblDrugInfo(
        phase=kw.get("phase", {"CHEMBL1": 4.0, "CHEMBL2": 4.0, "CHEMBL3": 1.0}),
        family=kw.get("family", {"PKIN": "kinase", "PGPCR": "gpcr", "P00001": "other"}),
        action_classes=_ACTION_CLASSES,
        binder_directions=kw.get("binder_directions", {}),
    )


def _gene(**overrides) -> pd.Series:
    base = {
        "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1", "gene_entrez_id": 111,
        "gene_uniprot_id": "P00001", "eqtl_source": "eqtlgen", "mr_beta": 0.5,
        "mr_pval": 1e-10, "pp_h4": 0.95, "confidence_tier": "high",
    }
    base.update(overrides)
    return pd.Series(base)


def _row(**kw) -> dict:
    row = {
        "gene_ensembl_id": "ENSG001", "gene_symbol": "GENE1", "gene_entrez_id": 111,
        "gene_uniprot_id": "P00001", "drug_chembl_id": "CHEMBL1", "drug_name": "DRUG",
        "interaction_type": "other", "pchembl_value": np.nan, "max_phase": 4,
        "mechanism_of_action": None, "source": "chembl", "confidence": "medium",
        "target_route": "none", "mechanism_action": None, "mechanism_direction": None,
        "dgidb_sources": None,
    }
    row.update(kw)
    return row


def _match(rows: list[dict], gene: pd.Series | None = None, info=None, **cfg) -> pd.DataFrame:
    config = MRDrugMatchConfig(match_mode="strict", min_phase=1, phase_filter_scope="chembl_only", **cfg)
    return match_drugs_to_mr_gene(gene if gene is not None else _gene(), pd.DataFrame(rows), config,
                                  info if info is not None else _info())


def _one(rows, **kw) -> pd.Series:
    m = _match(rows, **kw)
    assert len(m) == 1, m
    return m.iloc[0]


class TestCuratedRoutes:
    def test_single_protein_inhibitor_on_risk_gene_is_main_candidate(self) -> None:
        r = _one([_row(target_route="single_protein", mechanism_direction="negative")])
        assert (r["match_list"], r["match_status"], r["drug_direction"]) == ("main", "candidate", "negative")
        assert r["evidence_class"] == "curated_mechanism"

    def test_ace_inhibitor_is_predicted_adverse(self) -> None:
        # Criterion A5: lower ACE expression raises risk, so inhibition acts with it.
        r = _one([_row(target_route="single_protein", mechanism_direction="negative",
                       drug_name="CAPTOPRIL")], gene=_gene(gene_symbol="ACE", mr_beta=-0.56))
        assert r["match_status"] == "predicted_adverse"

    def test_member_routes_go_to_the_second_list(self) -> None:
        for route in ("site_member", "complex_member", "family_member"):
            r = _one([_row(target_route=route, mechanism_direction="positive")], gene=_gene(mr_beta=-0.3))
            assert (r["match_list"], r["match_status"]) == ("second", "candidate")

    def test_off_site_member_is_listed_and_never_directed(self) -> None:
        r = _one([_row(target_route="off_site", mechanism_direction="none")])
        assert (r["match_list"], r["match_status"], r["drug_direction"]) == ("listed", "off_site", "none")

    def test_nucleic_acid_target_is_main(self) -> None:
        r = _one([_row(target_route="nucleic_acid", mechanism_direction="negative")])
        assert (r["match_list"], r["match_status"]) == ("main", "candidate")

    def test_undirected_mechanism_is_a_binder(self) -> None:
        r = _one([_row(target_route="single_protein", mechanism_direction="none")])
        assert r["match_status"] == "binder"

    def test_mhc_and_steiger_still_apply(self) -> None:
        row = [_row(target_route="single_protein", mechanism_direction="negative")]
        assert _one(row, gene=_gene(mhc_flag=True))["match_status"] == "mhc_region"
        assert _one(row, gene=_gene(steiger_invalid_all_k=True))["match_status"] == "steiger_invalid"


class TestDgidbClaims:
    def test_typed_target_claim_on_approved_drug_can_be_a_candidate(self) -> None:
        r = _one([_row(source="dgidb", max_phase=0, confidence="low",
                       dgidb_sources=["GuideToPharmacology:antagonist", "PharmGKB:"])])
        assert (r["evidence_class"], r["match_list"], r["match_status"]) == ("dgidb_target", "main", "candidate")

    def test_target_claim_row_shows_the_chembl_phase_it_was_judged_on(self) -> None:
        r = _one([_row(drug_chembl_id="CHEMBL3", source="dgidb", max_phase=0,
                       dgidb_sources=["GuideToPharmacology:inhibitor"])])
        assert (r["match_status"], r["max_phase"]) == ("candidate", 1.0)
        kept = _one([_row(drug_chembl_id="CHEMBL9", source="dgidb", max_phase=0,
                          dgidb_sources=["GuideToPharmacology:inhibitor"])])
        assert kept["max_phase"] == 0

    def test_list_column_from_parquet_is_read(self) -> None:
        r = _one([_row(source="dgidb", max_phase=0,
                       dgidb_sources=np.array(["ChEMBL:inhibitor"], dtype=object))])
        assert r["match_status"] == "candidate"

    def test_without_chembl_phase_a_target_claim_is_low_evidence(self) -> None:
        rows = [_row(drug_chembl_id="CHEMBL9", source="dgidb", max_phase=0,
                     dgidb_sources=["GuideToPharmacology:antagonist"])]
        assert _one(rows)["match_status"] == "low_evidence"
        placeholder = [_row(drug_chembl_id="DGIDB_X", source="dgidb", max_phase=0,
                            dgidb_sources=["GuideToPharmacology:antagonist"])]
        assert _one(placeholder)["match_status"] == "low_evidence"

    def test_phase_floor_applies_through_chembl(self) -> None:
        rows = [_row(drug_chembl_id="CHEMBL3", source="dgidb", max_phase=0,
                     dgidb_sources=["TTD:inhibitor"])]
        config = dict(match_mode="strict", min_phase=4, phase_filter_scope="chembl_only")
        m = match_drugs_to_mr_gene(_gene(), pd.DataFrame(rows), MRDrugMatchConfig(**config), _info())
        assert m.iloc[0]["match_status"] == "low_evidence"

    def test_non_target_claims_are_listed(self) -> None:
        # Criterion A6.1: pharmacogenomic and biomarker claims never carry a candidate.
        r = _one([_row(source="dgidb", max_phase=0, interaction_type="inhibitor", confidence="medium",
                       dgidb_sources=["PharmGKB:", "MyCancerGenome:inhibitor"])])
        assert (r["evidence_class"], r["match_list"], r["match_status"]) == ("non_target", "listed", "low_evidence")

    def test_antibody_claim_has_no_direction(self) -> None:
        r = _one([_row(source="dgidb", max_phase=0, dgidb_sources=["GuideToPharmacology:antibody"])])
        assert (r["drug_direction"], r["match_status"]) == ("none", "binder")

    def test_disagreeing_types_have_no_direction(self) -> None:
        assert dgidb_direction(["inhibitor|agonist"], _ACTION_CLASSES) == "none"
        assert dgidb_direction(["inhibitor", "blocker"], _ACTION_CLASSES) == "negative"
        assert dgidb_direction(["partial agonist"], _ACTION_CLASSES) == "positive"

    def test_dtc_claim_without_potency_is_listed(self) -> None:
        r = _one([_row(source="dgidb", max_phase=0, dgidb_sources=["DTC:"])])
        assert (r["evidence_class"], r["match_list"], r["match_status"]) == ("binding", "listed", "low_evidence")


class TestBinders:
    def _affinity(self, uniprot: str, pch: float, **kw) -> dict:
        return _row(gene_uniprot_id=uniprot, pchembl_value=pch, **kw)

    def test_family_floors(self) -> None:
        assert _match([self._affinity("PKIN", 7.0)]).empty
        assert _one([self._affinity("PKIN", 8.0)])["match_list"] == "binder"
        assert _match([self._affinity("PGPCR", 6.5)]).empty
        assert _one([self._affinity("P00001", 6.5)])["match_status"] == "binder"

    def test_floor_options(self) -> None:
        assert _one([self._affinity("PGPCR", 6.5)], binder_floor=None)["match_list"] == "binder"
        assert _one([self._affinity("PGPCR", 6.5)], binder_floor=6.0)["match_list"] == "binder"
        assert _match([self._affinity("P00001", 6.5)], binder_floor=7.0).empty

    def test_curated_mechanism_is_matched_at_any_potency(self) -> None:
        r = _one([_row(gene_uniprot_id="PKIN", pchembl_value=5.0, target_route="single_protein",
                       mechanism_direction="negative")])
        assert r["match_status"] == "candidate"

    def test_pdsp_ki_is_a_binder(self) -> None:
        assert _one([self._affinity("P00001", 7.0, source="pdsp", max_phase=0)])["match_list"] == "binder"

    def test_directed_binder_only_when_its_class_is_enabled(self) -> None:
        info = _info(binder_directions={("CHEMBL1", "PKIN"): {"curated_activity": "negative",
                                                               "inferred_kinase_binding": "positive"}})
        row = [self._affinity("PKIN", 8.0)]
        assert _one(row, info=info)["match_list"] == "binder"
        r = _one(row, info=info, binder_direction_classes=["curated_activity"])
        assert (r["match_list"], r["match_status"], r["direction_source"]) == ("second", "candidate", "curated_activity")
        r = _one(row, info=info, binder_direction_classes=["inferred_kinase_binding"])
        assert (r["drug_direction"], r["direction_source"]) == ("positive", "inferred_kinase_binding")


class TestVerdictsAndSummaries:
    def test_verdict_reads_all_lists_but_the_second(self) -> None:
        matches = pd.DataFrame({"match_status": ["binder", "candidate"], "match_list": ["main", "second"]})
        v = _summarise_gene_verdict(_gene(), raw_count=2, matches=matches, config=MRDrugMatchConfig())
        assert v["verdict_status"] == "binder_only"
        assert v["second_list_status"] == "candidate"
        assert v["n_second_list_candidates"] == 1

    def test_compound_summary_marks_mixed(self) -> None:
        m = pd.DataFrame({
            "drug_chembl_id": ["CHEMBL1", "CHEMBL1", "CHEMBL2"],
            "drug_name": ["A", "A", "B"],
            "gene_ensembl_id": ["G1", "G2", "G1"], "gene_symbol": ["X", "Y", "X"],
            "eqtl_source": ["eqtlgen"] * 3, "match_list": ["main"] * 3,
            "match_status": ["candidate", "predicted_adverse", "binder"],
        })
        s = summarise_drug_compounds(m).set_index("drug_chembl_id")
        assert s.loc["CHEMBL1", "compound_summary"] == "mixed"
        assert s.loc["CHEMBL2", "compound_summary"] == "undirected"

    def test_primary_matches_leave_out_second_list_and_off_site(self) -> None:
        m = pd.DataFrame({"match_list": ["main", "second", "listed", "binder"],
                          "match_status": ["candidate", "candidate", "off_site", "binder"]})
        assert list(primary_mr_matches(m)["match_list"]) == ["main", "binder"]


class TestEligibleAndCalibration:
    def _results(self) -> pd.DataFrame:
        ids = [f"ENSG{i:03d}" for i in range(20)]
        return pd.DataFrame({
            "gene_ensembl_id": ids, "gene_symbol": [f"G{i}" for i in range(20)],
            "gene_entrez_id": range(20), "gene_uniprot_id": [f"P{i}" for i in range(20)],
            "eqtl_source": "eqtlgen", "mr_beta": 0.5, "mr_pval": 1e-12,
            "n_instruments": [1] * 20, "pp_h4": 0.9, "confidence_tier": "high",
        })

    def test_routes_rows_are_matched_beside_the_table(self) -> None:
        res = self._results().iloc[:2]
        table = pd.DataFrame([_row(gene_ensembl_id="ENSG000", gene_symbol="G0", gene_uniprot_id="P0",
                                   gene_entrez_id=0, target_route="single_protein",
                                   mechanism_direction="negative")])
        routes = pd.DataFrame([_row(gene_ensembl_id="ENSG001", gene_symbol="G1", gene_uniprot_id="P1",
                                    gene_entrez_id=1, drug_chembl_id="CHEMBL2",
                                    target_route="nucleic_acid", mechanism_direction="negative")])
        config = MRDrugMatchConfig(match_mode="strict", min_phase=1, phase_filter_scope="chembl_only")
        m, v = match_drugs_for_eligible(res, table, config, routes=routes, chembl_info=_info(),
                                        return_verdicts=True)
        assert set(zip(m["gene_symbol"], m["target_route"])) == {("G0", "single_protein"), ("G1", "nucleic_acid")}
        assert list(v["verdict_status"]) == ["candidate", "candidate"]

    def test_calibration_is_seeded_and_counts_calls(self) -> None:
        res = self._results()
        eligible = res.iloc[:4]
        table = pd.DataFrame({"gene_ensembl_id": res["gene_ensembl_id"].iloc[:10],
                              "target_route": "single_protein", "max_phase": 4})
        matches = pd.DataFrame({
            "drug_chembl_id": ["D1", "D2", "D2"], "gene_ensembl_id": ["ENSG000", "ENSG001", "ENSG001"],
            "match_list": ["main", "main", "second"],
            "match_status": ["candidate", "predicted_adverse", "candidate"],
        })
        a = drug_layer_calibration(res, eligible, table, matches, n_perm=500)
        b = drug_layer_calibration(res, eligible, table, matches, n_perm=500)
        assert a == b
        share = a["drugged_share"][0]
        assert share["share"] == 1.0 and share["null_mean"] == pytest.approx(0.5, abs=0.05)
        assert share["p_one_sided"] < 0.1
        assert a["directed_calls"]["candidate_pairs"] == 1
        assert a["directed_calls"]["predicted_adverse_pairs"] == 1
