"""Tests for repogen.data.gwas_prep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repogen.data.gwas_prep import (
    GWASFormat,
    apply_liftover,
    detect_gwas_format,
    harmonize_alleles,
    map_columns,
    prepare_gwas,
    prepare_gwas_for_spredixcan,
    quality_control,
)


# ---------------------------------------------------------------------------
# Helpers for creating synthetic GWAS files
# ---------------------------------------------------------------------------


def _write_pgc_gwas(path: Path, n: int = 20) -> None:
    """Write a PGC-style tab-separated GWAS file."""
    lines = ["SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP\tN"]
    for i in range(1, n + 1):
        lines.append(
            f"rs{i}\t{(i % 22) + 1}\t{i * 10000}\tA\tG\t"
            f"{0.05 * ((-1) ** i)}\t0.01\t{min(i * 0.002, 0.999)}\t50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_ukb_gwas(path: Path, n: int = 10) -> None:
    """Write a UKB-style space-separated GWAS file with VARIANT_ID."""
    lines = ["VARIANT_ID BETA SE P N"]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        pos = i * 10000
        lines.append(
            f"{chrom}:{pos}:A:G 0.03 0.01 {min(i * 0.005, 0.999)} 300000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_or_gwas(path: Path, n: int = 10) -> None:
    """Write a GWAS file with OR instead of BETA."""
    lines = ["SNP\tCHR\tBP\tA1\tA2\tOR\tSE\tP\tN"]
    for i in range(1, n + 1):
        lines.append(
            f"rs{i}\t{(i % 22) + 1}\t{i * 10000}\tA\tG\t"
            f"{1.0 + 0.05 * ((-1) ** i)}\t0.01\t{min(i * 0.002, 0.999)}\t20000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_case_control_gwas(path: Path, n: int = 10) -> None:
    """Write a GWAS file with N_CAS and N_CON but no N."""
    lines = ["SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP\tN_CAS\tN_CON"]
    for i in range(1, n + 1):
        lines.append(
            f"rs{i}\t{(i % 22) + 1}\t{i * 10000}\tA\tG\t"
            f"0.05\t0.01\t{min(i * 0.003, 0.999)}\t10000\t40000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_finngen_gwas(path: Path, n: int = 10) -> None:
    """Write a FinnGen-style file with #chrom header."""
    lines = ["#chrom\tpos\tref\talt\trsids\tbeta\tsebeta\tpval\tmaf\tmaf_cases\tmaf_controls"]
    for i in range(1, n + 1):
        lines.append(
            f"chr{(i % 22) + 1}\t{i * 10000}\tA\tG\trs{i}\t0.03\t0.01\t"
            f"{min(i * 0.004, 0.999)}\t0.15\t0.16\t0.14"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_csv_gwas(path: Path, n: int = 10) -> None:
    """Write a comma-separated GWAS file."""
    lines = ["SNP,CHR,POS,A1,A2,BETA,SE,P,N"]
    for i in range(1, n + 1):
        lines.append(
            f"rs{i},{(i % 22) + 1},{i * 10000},A,G,"
            f"0.05,0.01,{min(i * 0.002, 0.999)},50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_pgc_whitespace_gwas(path: Path, n: int = 20) -> None:
    """Write a PGC-style whitespace-delimited file with OR, Nca, Nco, Neff.

    Uses imbalanced case/control counts to produce fractional Neff values,
    matching real PGC data where Neff is rarely an exact integer.
    """
    lines = ["CHR SNP BP A1 A2 OR SE P Nca Nco Neff_half Neff"]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        nca, nco = 456419, 2778695
        neff = round(4 * nca * nco / (nca + nco), 1)
        lines.append(
            f"{chrom} rs{i} {i * 10000} A G "
            f"{1.0 + 0.02 * ((-1) ** i)} 0.01 {min(i * 0.002, 0.999)} "
            f"{nca} {nco} {neff / 2} {neff}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_metal_marker_gwas(path: Path, ref_bim: pd.DataFrame, n: int = 20) -> None:
    """Write a METAL-style file with CHR:POS_A1_A2 MarkerName (no rsIDs).

    Uses coordinates from *ref_bim* so that a coordinate-based fallback will
    find matches.  A few extra rows with no BIM match are appended for coverage.
    """
    header = "MarkerName\tCHR\tBP\tAllele1\tAllele2\tEffect\tStdErr\tP-value\tN"
    lines = [header]
    for i, (_, row) in enumerate(ref_bim.head(n).iterrows()):
        chrom = int(row.iloc[0])
        pos = int(row.iloc[3])
        ref_allele = str(row.iloc[4]).upper()
        alt_allele = str(row.iloc[5]).upper()
        marker = f"{chrom}:{pos}_{alt_allele}_{ref_allele}"
        beta = round(0.03 * ((-1) ** i), 4)
        lines.append(
            f"{marker}\t{chrom}\t{pos}\t{alt_allele.lower()}\t{ref_allele.lower()}\t"
            f"{beta}\t0.01\t{min((i + 1) * 0.002, 0.999)}\t50000"
        )
    for j in range(3):
        lines.append(
            f"99:{99000 + j}_X_Y\t99\t{99000 + j}\tx\ty\t0.01\t0.01\t0.5\t50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_vcf_preamble_gwas(path: Path, n: int = 20) -> None:
    """Write a PGCsumstatsVCF-style file with ## metadata preamble."""
    import gzip as _gz

    meta = [
        '##fileFormat=PGCsumstatsVCFv1.0',
        '##CAVEAT EMPTOR: ALWAYS CHECK FOR NEWER VERSION',
        '##genomeReference="GRCh37"',
        '##dependentVariableType="discrete"',
        '##nCase="52017"',
        '##nControl="75889"',
    ]
    header = "CHROM\tID\tPOS\tA1\tA2\tFCAS\tFCON\tIMPINFO\tBETA\tSE\tPVAL\tNCAS\tNCON\tNEFF"
    lines = meta + [header]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        lines.append(
            f"{chrom}\trs{i}\t{i * 10000}\tA\tG\t"
            f"0.15\t0.14\t0.95\t{0.05 * ((-1) ** i)}\t0.01\t"
            f"{min(i * 0.002, 0.999)}\t53386\t77258\t58749.13"
        )
    with _gz.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _write_vcf_hash_header_gwas(path: Path, n: int = 20) -> None:
    """Write a VCF-style file where the header starts with #CHROM."""
    meta = [
        '##fileformat=VCFv4.1',
        '##INFO=<ID=BETA,Number=1,Type=Float>',
    ]
    header = "#CHROM\tID\tPOS\tA1\tA2\tBETA\tSE\tPVAL\tN"
    lines = meta + [header]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        lines.append(
            f"{chrom}\trs{i}\t{i * 10000}\tA\tG\t"
            f"{0.05 * ((-1) ** i)}\t0.01\t{min(i * 0.002, 0.999)}\t50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_hash_header_no_preamble_gwas(path: Path, n: int = 20) -> None:
    """Write a file with #CHROM header but NO ## metadata preamble."""
    header = "#CHROM\tID\tPOS\tA1\tA2\tBETA\tSE\tPVAL\tN"
    lines = [header]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        lines.append(
            f"{chrom}\trs{i}\t{i * 10000}\tA\tG\t"
            f"{0.05 * ((-1) ** i)}\t0.01\t{min(i * 0.002, 0.999)}\t50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_hash_comment_plus_normal_header_gwas(path: Path, n: int = 20) -> None:
    """Write a file with a leading # comment line followed by a normal header."""
    lines = [
        "# This is a comment line, not a real header",
        "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP\tN",
    ]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        lines.append(
            f"rs{i}\t{chrom}\t{i * 10000}\tA\tG\t"
            f"{0.05 * ((-1) ** i)}\t0.01\t{min(i * 0.002, 0.999)}\t50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_bmi_giant_gwas(path: Path, n: int = 20) -> None:
    """Write a GIANT BMI-style file with Tested_Allele, rs:A:B SNP IDs."""
    header = "CHR POS SNP Tested_Allele Other_Allele Freq_Tested_Allele BETA SE P N INFO"
    lines = [header]
    alleles = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        pos = i * 10000
        a1, a2 = alleles[i % len(alleles)]
        lines.append(
            f"{chrom} {pos} rs{i}:{a1}:{a2} {a1.lower()} {a2.lower()} "
            f"0.15 {0.03 * ((-1) ** i):.4f} 0.01 {min(i * 0.002, 0.999)} 484680 0.9"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_variable_whitespace_gwas(path: Path, n: int = 10) -> None:
    """Write a column-aligned file with variable-width whitespace."""
    header = "CHR  SNP         BP       A1 A2 BETA   SE     P       N"
    lines = [header]
    for i in range(1, n + 1):
        chrom = (i % 22) + 1
        lines.append(
            f"{chrom:>3}  rs{i:<9} {i * 10000:<8} A  G  "
            f"0.05   0.01   {min(i * 0.002, 0.999):<7.3f} 50000"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_mixed_whitespace_gwas(path: Path) -> None:
    """Write a file where early lines are single-space but a later line has repeated spaces.

    Used to verify that the adaptive delimiter check catches anomalies
    anywhere within the detection window, not just on the first data line.
    """
    lines = [
        "CHR SNP BP A1 A2 BETA SE P N",
        "1 rs1 10000 A G 0.05 0.01 0.002 50000",
        "2 rs2 20000 C T 0.03 0.02 0.004 50000",
        "3 rs3 30000 A G  0.01 0.02 0.006  50000",  # repeated spaces
        "4 rs4 40000 G T 0.02 0.01 0.008 50000",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_bmi_with_malformed_tail(path: Path, n_good: int = 20) -> None:
    """Write a BMI-style file where the last rows have NA for CHR, POS, N, INFO.

    Mimics the real GIANT BMI 2018 file where ~1,050 tail rows are malformed.
    """
    header = "CHR POS SNP Tested_Allele Other_Allele Freq_Tested_Allele BETA SE P N INFO"
    lines = [header]
    alleles = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")]
    for i in range(1, n_good + 1):
        chrom = (i % 22) + 1
        pos = i * 10000
        a1, a2 = alleles[i % len(alleles)]
        lines.append(
            f"{chrom} {pos} rs{i}:{a1}:{a2} {a1.lower()} {a2.lower()} "
            f"0.15 {0.03 * ((-1) ** i):.4f} 0.01 {min(i * 0.002, 0.999)} 484680 0.9"
        )
    lines.append("NA NA rs9999:T:C t c 0.74 0.001 0.004 0.79 NA NA")
    lines.append("NA NA rs9998:A:G a g 0.50 0.002 0.005 0.65 NA NA")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Tests: detect_gwas_format
# ---------------------------------------------------------------------------


class TestDetectGwasFormat:
    """Tests for GWAS format auto-detection."""

    def test_pgc_format(self, tmp_path: Path) -> None:
        f = tmp_path / "pgc.txt"
        _write_pgc_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.source == "PGC"
        assert fmt.delimiter == "\t"
        assert fmt.has_beta is True
        assert fmt.has_or is False
        assert "SNP" in fmt.column_map
        assert "P" in fmt.column_map

    def test_ukb_format(self, tmp_path: Path) -> None:
        f = tmp_path / "ukb.txt"
        _write_ukb_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.source == "UKB"
        assert fmt.has_variant_id is True

    def test_or_detection(self, tmp_path: Path) -> None:
        f = tmp_path / "or_gwas.txt"
        _write_or_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.has_or is True
        assert fmt.has_beta is False
        assert "OR" in fmt.column_map

    def test_csv_delimiter(self, tmp_path: Path) -> None:
        f = tmp_path / "gwas.csv"
        _write_csv_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.delimiter == ","

    def test_finngen_format(self, tmp_path: Path) -> None:
        f = tmp_path / "finngen.txt"
        _write_finngen_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.source == "FinnGen"

    def test_sample_size_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "pgc.txt"
        _write_pgc_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.has_sample_size is True

    def test_case_control_sample_size(self, tmp_path: Path) -> None:
        f = tmp_path / "cc.txt"
        _write_case_control_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.has_sample_size is True

    def test_pgc_whitespace_nca_nco_detected(self, tmp_path: Path) -> None:
        """Nca/Nco columns from PGC whitespace files are detected."""
        f = tmp_path / "pgc_ws.txt"
        _write_pgc_whitespace_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.delimiter == " "
        assert fmt.has_sample_size is True
        assert "N_CAS" in fmt.column_map
        assert "N_CON" in fmt.column_map
        assert fmt.column_map["N_CAS"] == "Nca"
        assert fmt.column_map["N_CON"] == "Nco"

    def test_vcf_preamble_skips_metadata(self, tmp_path: Path) -> None:
        """## metadata preamble is skipped; real header detected."""
        f = tmp_path / "pgc_vcf.tsv.gz"
        _write_vcf_preamble_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.n_meta_lines == 6
        assert fmt.delimiter == "\t"
        assert "P" in fmt.column_map
        assert "CHR" in fmt.column_map
        assert "SNP" in fmt.column_map
        assert fmt.column_map["P"] == "PVAL"
        assert fmt.column_map["CHR"] == "CHROM"
        assert fmt.column_map["SNP"] == "ID"
        assert fmt.has_sample_size is True

    def test_vcf_preamble_genome_build(self, tmp_path: Path) -> None:
        """##genomeReference is extracted as genome_build during detection."""
        f = tmp_path / "pgc_vcf.tsv.gz"
        _write_vcf_preamble_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.genome_build == "GRCh37"

    def test_vcf_hash_header_detected(self, tmp_path: Path) -> None:
        """#CHROM-style header (after ## lines) is correctly detected."""
        f = tmp_path / "vcf.txt"
        _write_vcf_hash_header_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.n_meta_lines == 2
        assert "CHR" in fmt.column_map
        assert fmt.source == "FinnGen"

    def test_hash_header_no_preamble_detected(self, tmp_path: Path) -> None:
        """#CHROM header with no ## preamble: detection succeeds and sets flag."""
        f = tmp_path / "no_preamble.txt"
        _write_hash_header_no_preamble_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.n_meta_lines == 0
        assert fmt.header_starts_with_hash is True
        assert "CHR" in fmt.column_map
        assert "P" in fmt.column_map
        assert fmt.source == "FinnGen"

    def test_hash_comment_line_not_flagged(self, tmp_path: Path) -> None:
        """Leading # comment (not a column header) does NOT set header_starts_with_hash.

        Detection treats the # comment as the header (0 columns mapped).
        The read step handles this via comment='#' + fallback alias mapping.
        """
        f = tmp_path / "comment.txt"
        _write_hash_comment_plus_normal_header_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.header_starts_with_hash is False
        assert len(fmt.column_map) == 0

    def test_bmi_giant_tested_allele_detected(self, tmp_path: Path) -> None:
        """GIANT BMI Tested_Allele -> A1 and Other_Allele -> A2 are detected."""
        f = tmp_path / "bmi.txt"
        _write_bmi_giant_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.delimiter == " "
        assert "A1" in fmt.column_map
        assert "A2" in fmt.column_map
        assert fmt.column_map["A1"] == "Tested_Allele"
        assert fmt.column_map["A2"] == "Other_Allele"
        assert "MAF" in fmt.column_map
        assert fmt.column_map["MAF"] == "Freq_Tested_Allele"

    def test_variable_whitespace_keeps_regex(self, tmp_path: Path) -> None:
        """Column-aligned file with repeated spaces retains \\s+ delimiter."""
        f = tmp_path / "aligned.txt"
        _write_variable_whitespace_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.delimiter == r"\s+"

    def test_mixed_whitespace_keeps_regex(self, tmp_path: Path) -> None:
        """File where a later sampled line has repeated spaces keeps \\s+."""
        f = tmp_path / "mixed.txt"
        _write_mixed_whitespace_gwas(f)
        fmt = detect_gwas_format(f)
        assert fmt.delimiter == r"\s+"


# ---------------------------------------------------------------------------
# Tests: map_columns
# ---------------------------------------------------------------------------


class TestMapColumns:
    """Tests for column mapping and transforms."""

    def test_renames_columns(self) -> None:
        fmt = GWASFormat(
            column_map={"SNP": "RSID", "P": "PVAL", "BETA": "B"},
        )
        df = pd.DataFrame({"RSID": ["rs1"], "PVAL": [0.01], "B": [0.5]})
        result = map_columns(df, fmt)
        assert "SNP" in result.columns
        assert "P" in result.columns
        assert "BETA" in result.columns

    def test_or_to_beta_conversion(self) -> None:
        fmt = GWASFormat(has_or=True, column_map={"OR": "OR"})
        df = pd.DataFrame({"OR": [2.0, 0.5]})
        result = map_columns(df, fmt)
        assert "BETA" in result.columns
        assert abs(result["BETA"].iloc[0] - np.log(2.0)) < 1e-6
        assert abs(result["BETA"].iloc[1] - np.log(0.5)) < 1e-6

    def test_n_from_cases_controls(self) -> None:
        fmt = GWASFormat(
            column_map={"N_CAS": "N_CAS", "N_CON": "N_CON"},
        )
        df = pd.DataFrame({"N_CAS": [1000], "N_CON": [4000]})
        result = map_columns(df, fmt)
        assert "N" in result.columns
        assert result["N"].iloc[0] == 5000

    def test_variant_id_parsing(self) -> None:
        fmt = GWASFormat(
            has_variant_id=True,
            column_map={"VARIANT_ID": "VARIANT_ID"},
        )
        df = pd.DataFrame({"VARIANT_ID": ["1:100000:A:G"]})
        result = map_columns(df, fmt)
        assert result["CHR"].iloc[0] == 1
        assert result["POS"].iloc[0] == 100000
        assert result["A1"].iloc[0] == "A"
        assert result["A2"].iloc[0] == "G"

    def test_variant_id_construction(self) -> None:
        fmt = GWASFormat(
            column_map={"CHR": "CHR", "POS": "POS", "A1": "A1", "A2": "A2"},
        )
        df = pd.DataFrame({"CHR": [1], "POS": [100000], "A1": ["a"], "A2": ["g"]})
        result = map_columns(df, fmt)
        assert "VARIANT_ID" in result.columns
        assert result["VARIANT_ID"].iloc[0] == "1:100000:A:G"

    def test_allele_uppercasing(self) -> None:
        fmt = GWASFormat(column_map={"A1": "A1", "A2": "A2"})
        df = pd.DataFrame({"A1": ["a", "t"], "A2": ["g", "c"]})
        result = map_columns(df, fmt)
        assert result["A1"].tolist() == ["A", "T"]
        assert result["A2"].tolist() == ["G", "C"]

    def test_chr_prefix_stripped(self) -> None:
        fmt = GWASFormat(column_map={"CHR": "CHR"})
        df = pd.DataFrame({"CHR": ["chr1", "chr22"]})
        result = map_columns(df, fmt)
        assert result["CHR"].tolist() == [1, 22]

    def test_snp_column_added_when_missing(self) -> None:
        fmt = GWASFormat(column_map={"P": "P"})
        df = pd.DataFrame({"P": [0.01]})
        result = map_columns(df, fmt)
        assert "SNP" in result.columns
        assert pd.isna(result["SNP"].iloc[0])

    def test_fallback_alias_mapping(self) -> None:
        """Columns unmapped by detection are still aliased via GWAS_COLUMN_ALIASES."""
        fmt = GWASFormat(column_map={})
        df = pd.DataFrame({
            "PVAL": [0.01], "CHROM": [1], "ID": ["rs1"],
            "BETA": [0.05], "SE": [0.01],
        })
        result = map_columns(df, fmt)
        assert "P" in result.columns
        assert "CHR" in result.columns
        assert "SNP" in result.columns

    def test_snp_rs_suffix_normalised(self) -> None:
        """SNP IDs like rs123:C:A are normalised to rs123."""
        fmt = GWASFormat(column_map={"SNP": "SNP"})
        df = pd.DataFrame({
            "SNP": ["rs1:A:G", "rs2:C:T", "rs3"],
            "CHR": [1, 1, 1],
            "POS": [100, 200, 300],
            "A1": ["A", "C", "A"],
            "A2": ["G", "T", "G"],
            "BETA": [0.1, 0.2, 0.3],
            "SE": [0.01, 0.01, 0.01],
            "P": [0.05, 0.01, 0.001],
        })
        result = map_columns(df, fmt)
        assert result["SNP"].tolist() == ["rs1", "rs2", "rs3"]

    def test_snp_plain_rsid_unchanged(self) -> None:
        """Plain rsIDs without suffixes are not modified."""
        fmt = GWASFormat(column_map={"SNP": "SNP"})
        df = pd.DataFrame({
            "SNP": ["rs100", "rs200"],
            "BETA": [0.1, 0.2],
        })
        result = map_columns(df, fmt)
        assert result["SNP"].tolist() == ["rs100", "rs200"]


# ---------------------------------------------------------------------------
# Tests: quality_control
# ---------------------------------------------------------------------------


class TestQualityControl:
    """Tests for QC filtering."""

    def _make_df(self, n: int = 100) -> pd.DataFrame:
        return pd.DataFrame({
            "SNP": [f"rs{i}" for i in range(n)],
            "VARIANT_ID": [f"1:{i * 1000}:A:G" for i in range(n)],
            "CHR": [1] * n,
            "POS": [i * 1000 for i in range(n)],
            "A1": ["A"] * n,
            "A2": ["G"] * n,
            "BETA": np.random.randn(n) * 0.1,
            "SE": [0.01] * n,
            "P": np.random.uniform(0.0001, 0.999, n),
            "N": [50000] * n,
            "INFO": np.random.uniform(0.3, 1.0, n),
            "MAF": np.random.uniform(0.001, 0.5, n),
        })

    def test_info_filter(self) -> None:
        df = self._make_df(50)
        df.loc[:9, "INFO"] = 0.3
        result = quality_control(df, info_threshold=0.6, maf_threshold=0.0)
        assert len(result) < len(df)
        assert result["INFO"].min() >= 0.6

    def test_maf_filter(self) -> None:
        df = self._make_df(50)
        df.loc[:9, "MAF"] = 0.001
        result = quality_control(df, info_threshold=0.0, maf_threshold=0.01)
        assert len(result) < len(df)
        assert result["MAF"].min() >= 0.01

    def test_p_value_validation(self) -> None:
        df = self._make_df(20)
        df.loc[0, "P"] = -0.1
        df.loc[1, "P"] = 1.5
        df.loc[2, "P"] = np.nan
        result = quality_control(df, info_threshold=0.0, maf_threshold=0.0)
        assert len(result) == len(df) - 3

    def test_deduplication(self) -> None:
        df = self._make_df(10)
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        result = quality_control(df, info_threshold=0.0, maf_threshold=0.0)
        assert len(result) == 10

    def test_mhc_removal(self) -> None:
        df = self._make_df(20)
        df["CHR"] = 6
        df.loc[:9, "POS"] = 30_000_000
        df.loc[10:, "POS"] = 50_000_000
        result = quality_control(
            df, info_threshold=0.0, maf_threshold=0.0, remove_mhc=True,
        )
        assert len(result) == 10

    def test_no_info_column_ok(self) -> None:
        df = self._make_df(10).drop(columns=["INFO"])
        result = quality_control(df, info_threshold=0.6, maf_threshold=0.0)
        assert len(result) == 10

    def test_no_maf_column_ok(self) -> None:
        df = self._make_df(10).drop(columns=["MAF"])
        result = quality_control(df, info_threshold=0.0, maf_threshold=0.01)
        assert len(result) == 10

    def test_missing_p_column_raises_descriptive_error(self) -> None:
        """Missing P column raises ValueError, not KeyError from pandas."""
        df = pd.DataFrame({
            "SNP": ["rs1"], "CHR": [1], "POS": [1000],
            "A1": ["A"], "A2": ["G"], "BETA": [0.05], "SE": [0.01],
            "PVAL": [0.01],
        })
        with pytest.raises(ValueError, match="Required column 'P' not found"):
            quality_control(df)

    def test_malformed_rows_with_null_required_fields_dropped(self) -> None:
        """Rows with NaN in required fields (N, CHR, POS) are dropped by QC."""
        df = pd.DataFrame({
            "SNP": ["rs1", "rs2", "rs3"],
            "CHR": pd.array([1, 2, pd.NA], dtype="Int64"),
            "POS": pd.array([1000, 2000, pd.NA], dtype="Int64"),
            "A1": ["A", "C", "T"],
            "A2": ["G", "T", "C"],
            "BETA": [0.05, -0.03, 0.001],
            "SE": [0.01, 0.02, 0.004],
            "P": [0.01, 0.05, 0.79],
            "N": pd.array([50000, 50000, pd.NA], dtype="Int64"),
        })
        result = quality_control(df)
        assert len(result) == 2
        assert set(result["SNP"].tolist()) == {"rs1", "rs2"}

    def test_all_rows_complete_none_dropped(self) -> None:
        """Fully complete rows pass the required-field check untouched."""
        df = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "CHR": [1, 2],
            "POS": [1000, 2000],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "BETA": [0.05, -0.03],
            "SE": [0.01, 0.02],
            "P": [0.01, 0.05],
            "N": [50000, 50000],
        })
        result = quality_control(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: harmonize_alleles
# ---------------------------------------------------------------------------


class TestHarmonizeAlleles:
    """Tests for allele harmonisation against reference."""

    def _make_ref_bim(self) -> pd.DataFrame:
        return pd.DataFrame({
            "CHR": [1, 1, 1, 1],
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "CM": [0, 0, 0, 0],
            "POS": [1000, 2000, 3000, 4000],
            "REF": ["A", "C", "A", "C"],
            "ALT": ["G", "T", "T", "G"],
        })

    def test_matching_alleles_kept(self) -> None:
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["rs1"],
            "A1": ["G"],
            "A2": ["A"],
            "BETA": [0.5],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 1
        assert result.iloc[0]["BETA"] == 0.5

    def test_flipped_alleles(self) -> None:
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["rs1"],
            "A1": ["A"],
            "A2": ["G"],
            "BETA": [0.5],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 1
        assert result.iloc[0]["BETA"] == -0.5

    def test_palindromic_removed(self) -> None:
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["rs3"],
            "A1": ["A"],
            "A2": ["T"],
            "BETA": [0.5],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 0

    def test_snp_not_in_ref_dropped(self) -> None:
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["rs999"],
            "A1": ["A"],
            "A2": ["G"],
            "BETA": [0.1],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 0

    def test_coordinate_fallback_when_snp_mismatch(self) -> None:
        """Non-rsID SNP column but valid CHR+POS: falls back to coordinate merge."""
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["1:1000_G_A", "1:2000_T_C"],
            "CHR": pd.array([1, 1], dtype="Int64"),
            "POS": pd.array([1000, 2000], dtype="Int64"),
            "A1": ["G", "T"],
            "A2": ["A", "C"],
            "BETA": [0.5, 0.3],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 2
        assert set(result["SNP"]) == {"rs1", "rs2"}
        assert result.loc[result["SNP"] == "rs1", "BETA"].iloc[0] == 0.5
        assert result.loc[result["SNP"] == "rs2", "BETA"].iloc[0] == 0.3

    def test_coordinate_fallback_flips_alleles(self) -> None:
        """Coordinate fallback still flips BETA when alleles are inverted."""
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["1:1000_A_G"],
            "CHR": pd.array([1], dtype="Int64"),
            "POS": pd.array([1000], dtype="Int64"),
            "A1": ["A"],
            "A2": ["G"],
            "BETA": [0.5],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 1
        assert result.iloc[0]["SNP"] == "rs1"
        assert result.iloc[0]["BETA"] == -0.5

    def test_coordinate_fallback_no_coord_match_raises_nothing(self) -> None:
        """Coordinate fallback yields 0 matches (build mismatch): returns empty."""
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["1:9999_G_A"],
            "CHR": pd.array([1], dtype="Int64"),
            "POS": pd.array([9999], dtype="Int64"),
            "A1": ["G"],
            "A2": ["A"],
            "BETA": [0.1],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 0

    def test_snp_merge_still_primary_path(self) -> None:
        """When rsIDs match, coordinate fallback is NOT activated (regression)."""
        ref = self._make_ref_bim()
        gwas = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "CHR": pd.array([1, 1], dtype="Int64"),
            "POS": pd.array([1000, 2000], dtype="Int64"),
            "A1": ["G", "T"],
            "A2": ["A", "C"],
            "BETA": [0.5, 0.7],
        })
        result = harmonize_alleles(gwas, ref)
        assert len(result) == 2
        assert set(result["SNP"]) == {"rs1", "rs2"}


# ---------------------------------------------------------------------------
# Tests: prepare_gwas (end-to-end)
# ---------------------------------------------------------------------------


class TestPrepareGwas:
    """End-to-end tests for the main entry point."""

    def test_pgc_format_end_to_end(self, tmp_path: Path) -> None:
        f = tmp_path / "pgc.txt"
        _write_pgc_gwas(f, n=50)
        df, meta = prepare_gwas(
            input_path=f,
            genome_build="GRCh37",
            trait_type="case_control",
        )
        assert len(df) > 0
        assert meta.genome_build == "GRCh37"
        assert meta.source == "PGC"
        assert "SNP" in df.columns
        assert "BETA" in df.columns
        assert "P" in df.columns
        assert "N" in df.columns

    def test_or_conversion_end_to_end(self, tmp_path: Path) -> None:
        f = tmp_path / "or_gwas.txt"
        _write_or_gwas(f, n=20)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37",
        )
        assert "BETA" in df.columns
        assert df["BETA"].notna().all()

    def test_case_control_n_computation(self, tmp_path: Path) -> None:
        f = tmp_path / "cc.txt"
        _write_case_control_gwas(f, n=20)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37",
        )
        assert "N" in df.columns
        assert (df["N"] == 50000).all()

    def test_sample_size_override(self, tmp_path: Path) -> None:
        f = tmp_path / "pgc.txt"
        _write_pgc_gwas(f, n=10)
        lines = f.read_text().split("\n")
        lines[0] = "SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP"
        rewritten = []
        for line in lines:
            if line.startswith("rs"):
                parts = line.split("\t")
                rewritten.append("\t".join(parts[:8]))
            else:
                rewritten.append(line)
        f.write_text("\n".join(rewritten) + "\n")

        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37", sample_size=99999,
        )
        assert (df["N"] == 99999).all()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_gwas(input_path=tmp_path / "nonexistent.txt")

    def test_empty_after_qc_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tP\tN\nrs1\t1\t100\tA\tG\t0.1\t0.01\t-5\t1000\n")
        with pytest.raises(RuntimeError, match="No variants survived"):
            prepare_gwas(input_path=f, genome_build="GRCh37")

    def test_metadata_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "pgc.txt"
        _write_pgc_gwas(f, n=20)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh38", trait_type="quantitative",
        )
        assert meta.genome_build == "GRCh38"
        assert meta.trait_type == "quantitative"
        assert meta.original_n_variants == 20
        assert meta.n_variants_after_qc == len(df)
        assert meta.liftover_applied is False

    def test_csv_format(self, tmp_path: Path) -> None:
        f = tmp_path / "gwas.csv"
        _write_csv_gwas(f, n=15)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37",
        )
        assert len(df) > 0
        assert "BETA" in df.columns

    def test_whitespace_delimited_no_parser_crash(self, tmp_path: Path) -> None:
        """Single-space-delimited files parse successfully via C engine."""
        f = tmp_path / "pgc_ws.txt"
        _write_pgc_whitespace_gwas(f, n=20)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37", trait_type="case_control",
        )
        assert len(df) > 0
        assert meta.source == "PGC"

    def test_pgc_nca_nco_mapped(self, tmp_path: Path) -> None:
        """Nca/Nco/Neff columns from PGC are mapped to canonical N_CAS/N_CON/N."""
        f = tmp_path / "pgc_ws.txt"
        _write_pgc_whitespace_gwas(f, n=20)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37", trait_type="case_control",
        )
        assert "N" in df.columns
        assert df["N"].notna().all()
        assert "N_CAS" in df.columns
        assert "N_CON" in df.columns

    def test_pgc_or_converted_to_beta(self, tmp_path: Path) -> None:
        """PGC whitespace file with OR column gets BETA via log-transform."""
        f = tmp_path / "pgc_ws.txt"
        _write_pgc_whitespace_gwas(f, n=20)
        df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37",
        )
        assert "BETA" in df.columns
        assert df["BETA"].notna().all()

    def test_vcf_preamble_end_to_end(self, tmp_path: Path) -> None:
        """Gzipped VCF-style file with ## preamble parses correctly end-to-end."""
        f = tmp_path / "pgc_vcf.tsv.gz"
        _write_vcf_preamble_gwas(f, n=20)
        df, meta = prepare_gwas(input_path=f)
        assert len(df) > 0
        assert "P" in df.columns
        assert "BETA" in df.columns
        assert "CHR" in df.columns
        assert "SNP" in df.columns
        assert "N" in df.columns
        assert df["P"].notna().all()
        assert meta.genome_build == "GRCh37"

    def test_vcf_hash_header_end_to_end(self, tmp_path: Path) -> None:
        """File with #CHROM-style header (after ##) preserves header correctly."""
        f = tmp_path / "vcf.txt"
        _write_vcf_hash_header_gwas(f, n=20)
        df, meta = prepare_gwas(input_path=f, genome_build="GRCh37")
        assert len(df) > 0
        assert "P" in df.columns
        assert "CHR" in df.columns

    def test_hash_header_no_preamble_end_to_end(self, tmp_path: Path) -> None:
        """#CHROM header with no ## metadata: comment='#' must not eat the header."""
        f = tmp_path / "no_preamble.txt"
        _write_hash_header_no_preamble_gwas(f, n=20)
        df, meta = prepare_gwas(input_path=f, genome_build="GRCh37")
        assert len(df) > 0
        assert "P" in df.columns
        assert "CHR" in df.columns
        assert "SNP" in df.columns
        assert df["P"].notna().all()

    def test_metal_marker_coordinate_fallback(self, tmp_path: Path) -> None:
        """METAL-style MarkerName (CHR:POS_A1_A2) with BIM: coordinate fallback works."""
        bim_df = pd.DataFrame({
            "CHR": [1, 1, 1, 2, 2],
            "SNP": ["rs10", "rs20", "rs30", "rs40", "rs50"],
            "CM": [0, 0, 0, 0, 0],
            "POS": [10000, 20000, 30000, 10000, 20000],
            "REF": ["A", "C", "A", "G", "T"],
            "ALT": ["G", "T", "T", "C", "A"],
        })
        bim_path = tmp_path / "ref.bim"
        bim_df.to_csv(bim_path, sep="\t", header=False, index=False)

        gwas_path = tmp_path / "cad.tsv"
        _write_metal_marker_gwas(gwas_path, ref_bim=bim_df, n=5)

        df, meta = prepare_gwas(
            input_path=gwas_path,
            reference_bim=bim_path,
            genome_build="GRCh37",
        )
        assert len(df) > 0
        assert df["SNP"].str.startswith("rs").all()
        assert "P" in df.columns
        assert "BETA" in df.columns

    def test_harmonise_precheck_missing_a1_raises(self, tmp_path: Path) -> None:
        """Missing A1 column with reference_bim raises a descriptive ValueError."""
        bim_path = tmp_path / "ref.bim"
        pd.DataFrame({
            "CHR": [1], "SNP": ["rs1"], "CM": [0],
            "POS": [1000], "REF": ["A"], "ALT": ["G"],
        }).to_csv(bim_path, sep="\t", header=False, index=False)

        gwas_path = tmp_path / "bad.tsv"
        gwas_path.write_text("SNP\tBETA\tSE\tP\nrs1\t0.1\t0.01\t0.05\n")

        with pytest.raises(ValueError, match="A1"):
            prepare_gwas(
                input_path=gwas_path,
                reference_bim=bim_path,
                genome_build="GRCh37",
            )

    def test_bmi_giant_end_to_end(self, tmp_path: Path) -> None:
        """GIANT BMI-style file (Tested_Allele, rs:A:G) processes end-to-end."""
        alleles = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")]
        bim_ref, bim_alt = [], []
        for i in range(1, 21):
            a1, a2 = alleles[i % len(alleles)]
            bim_ref.append(a2)
            bim_alt.append(a1)
        bim_df = pd.DataFrame({
            "CHR": [(i % 22) + 1 for i in range(1, 21)],
            "SNP": [f"rs{i}" for i in range(1, 21)],
            "CM": [0] * 20,
            "POS": [i * 10000 for i in range(1, 21)],
            "REF": bim_ref,
            "ALT": bim_alt,
        })
        bim_path = tmp_path / "ref.bim"
        bim_df.to_csv(bim_path, sep="\t", header=False, index=False)

        gwas_path = tmp_path / "bmi.txt"
        _write_bmi_giant_gwas(gwas_path, n=20)

        df, meta = prepare_gwas(
            input_path=gwas_path,
            reference_bim=bim_path,
            genome_build="GRCh37",
            trait_type="quantitative",
        )
        assert len(df) > 0
        assert df["SNP"].str.match(r"^rs\d+$").all()
        assert "A1" in df.columns
        assert "A2" in df.columns
        assert "BETA" in df.columns
        assert "MAF" in df.columns

    def test_bmi_malformed_tail_rows_dropped(self, tmp_path: Path) -> None:
        """Malformed tail rows (NA CHR/POS/N/INFO) are removed by QC."""
        alleles = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")]
        bim_ref, bim_alt, bim_snps = [], [], []
        for i in range(1, 21):
            a1, a2 = alleles[i % len(alleles)]
            bim_ref.append(a2)
            bim_alt.append(a1)
            bim_snps.append(f"rs{i}")
        bim_snps.extend(["rs9999", "rs9998"])
        bim_ref.extend(["C", "G"])
        bim_alt.extend(["T", "A"])
        bim_df = pd.DataFrame({
            "CHR": [(i % 22) + 1 for i in range(1, 23)],
            "SNP": bim_snps,
            "CM": [0] * 22,
            "POS": [i * 10000 for i in range(1, 23)],
            "REF": bim_ref,
            "ALT": bim_alt,
        })
        bim_path = tmp_path / "ref.bim"
        bim_df.to_csv(bim_path, sep="\t", header=False, index=False)

        gwas_path = tmp_path / "bmi_bad.txt"
        _write_bmi_with_malformed_tail(gwas_path, n_good=20)

        df, meta = prepare_gwas(
            input_path=gwas_path,
            reference_bim=bim_path,
            genome_build="GRCh37",
            trait_type="quantitative",
        )
        assert df["N"].notna().all(), "Malformed rows with NaN N should have been dropped"
        assert df["CHR"].notna().all(), "Malformed rows with NaN CHR should have been dropped"
        assert "rs9999" not in df["SNP"].values
        assert "rs9998" not in df["SNP"].values


# ---------------------------------------------------------------------------
# Liftover Correctness Tests
# ---------------------------------------------------------------------------


class TestApplyLiftover:
    """Tests for coordinate basis, CHR update, strand, and multi-mapping."""

    @staticmethod
    def _mock_liftover(mapping: dict):
        """Create a mock LiftOver object with a predefined coordinate mapping."""
        from unittest.mock import MagicMock, patch

        lo = MagicMock()

        def convert(chrom: str, pos: int):
            key = (chrom, pos)
            return mapping.get(key, [])

        lo.convert_coordinate = convert
        return lo

    def test_coordinate_basis_conversion(self) -> None:
        """GWAS POS is 1-based; pyliftover expects 0-based."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1], "POS": [100], "SNP": ["rs1"],
            "A1": ["A"], "A2": ["G"],
        })

        mapping = {
            ("chr1", 99): [("chr1", 199, "+", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert len(result) == 1
        assert result.iloc[0]["POS"] == 200
        assert counters["converted"] == 1

    def test_chr_updated_from_liftover_output(self) -> None:
        """CHR should be updated from liftover result, not kept as-is."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1], "POS": [100], "SNP": ["rs1"],
            "A1": ["A"], "A2": ["G"],
        })
        mapping = {
            ("chr1", 99): [("chr2", 199, "+", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, _ = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert result.iloc[0]["CHR"] == 2

    def test_minus_strand_allele_complement(self) -> None:
        """Minus-strand mappings should complement alleles."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1], "POS": [100], "SNP": ["rs1"],
            "A1": ["A"], "A2": ["G"],
        })
        mapping = {
            ("chr1", 99): [("chr1", 199, "-", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert result.iloc[0]["A1"] == "T"
        assert result.iloc[0]["A2"] == "C"
        assert counters["minus_strand_complemented"] == 1

    def test_multi_mapping_dropped(self) -> None:
        """Variants with multiple liftover targets should be dropped."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1, 2], "POS": [100, 200], "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"], "A2": ["G", "T"],
        })
        mapping = {
            ("chr1", 99): [("chr1", 199, "+", 1), ("chr3", 399, "+", 0.8)],
            ("chr2", 199): [("chr2", 299, "+", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert len(result) == 1
        assert result.iloc[0]["SNP"] == "rs2"
        assert counters["multimap_dropped"] == 1

    def test_unmapped_variants_removed_with_counter(self) -> None:
        """Unmapped variants are removed and counted."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1, 2], "POS": [100, 200], "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"], "A2": ["G", "T"],
        })
        mapping = {
            ("chr1", 99): [("chr1", 199, "+", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert len(result) == 1
        assert counters["unmapped"] == 1
        assert counters["converted"] == 1

    def test_variant_id_rebuilt_after_liftover(self) -> None:
        """VARIANT_ID should be rebuilt with new CHR:POS:A1:A2."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1], "POS": [100], "SNP": ["rs1"],
            "A1": ["A"], "A2": ["G"],
            "VARIANT_ID": ["1:100:A:G"],
        })
        mapping = {
            ("chr1", 99): [("chr5", 999, "+", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, _ = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert result.iloc[0]["VARIANT_ID"] == "5:1000:A:G"

    def test_mixed_plus_minus_strand_no_broadcast_crash(self) -> None:
        """Mixed +/- strand rows must not crash (broadcast regression)."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1, 2, 3, 4],
            "POS": [100, 200, 300, 400],
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "A1": ["A", "C", "G", "T"],
            "A2": ["G", "T", "A", "C"],
        })
        mapping = {
            ("chr1", 99): [("chr1", 199, "+", 1)],
            ("chr2", 199): [("chr2", 299, "-", 1)],
            ("chr3", 299): [("chr3", 399, "+", 1)],
            ("chr4", 399): [("chr4", 499, "-", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert len(result) == 4
        assert result.iloc[0]["A1"] == "A"
        assert result.iloc[0]["A2"] == "G"
        assert result.iloc[1]["A1"] == "G"
        assert result.iloc[1]["A2"] == "A"
        assert result.iloc[2]["A1"] == "G"
        assert result.iloc[2]["A2"] == "A"
        assert result.iloc[3]["A1"] == "A"
        assert result.iloc[3]["A2"] == "G"
        assert counters["minus_strand_complemented"] == 2
        assert counters["minus_strand_dropped"] == 0

    def test_minus_strand_invalid_allele_dropped_with_counter(self) -> None:
        """Minus-strand rows with non-standard alleles are dropped."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1, 2, 3],
            "POS": [100, 200, 300],
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "N", "C"],
            "A2": ["G", "T", "G"],
        })
        mapping = {
            ("chr1", 99): [("chr1", 199, "+", 1)],
            ("chr2", 199): [("chr2", 299, "-", 1)],
            ("chr3", 299): [("chr3", 399, "-", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        assert len(result) == 2
        assert set(result["SNP"].tolist()) == {"rs1", "rs3"}
        assert counters["minus_strand_complemented"] == 1
        assert counters["minus_strand_dropped"] == 1
        assert counters["converted"] == 2

    def test_row_accounting_consistency(self) -> None:
        """Counters should be internally consistent with row accounting."""
        from unittest.mock import patch

        df = pd.DataFrame({
            "CHR": [1, 2, 3, 4, 5],
            "POS": [100, 200, 300, 400, 500],
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "A1": ["A", "C", "N", "T", "G"],
            "A2": ["G", "T", "A", "C", "A"],
        })
        mapping = {
            ("chr1", 99): [("chr1", 199, "+", 1)],
            ("chr3", 299): [("chr3", 399, "-", 1)],
            ("chr4", 399): [("chr4", 499, "-", 1)],
            ("chr5", 499): [("chr5", 599, "+", 1)],
        }
        lo = self._mock_liftover(mapping)

        with patch("pyliftover.LiftOver", return_value=lo):
            result, counters = apply_liftover(df, Path("fake.chain"), "GRCh37", "GRCh38")

        n_before = 5
        expected_remaining = (
            n_before
            - counters["unmapped"]
            - counters["multimap_dropped"]
            - counters["minus_strand_dropped"]
            - counters.get("non_numeric_chr_dropped", 0)
        )
        assert counters["converted"] == expected_remaining
        assert counters["converted"] == len(result)
        assert counters["unmapped"] == 1
        assert counters["minus_strand_dropped"] == 1
        assert counters["minus_strand_complemented"] == 1


class TestPrepareGwasForSpredixcan:
    """Tests for Branch-B GWAS preparation from raw GWAS input."""

    def test_raw_gwas_produces_output_with_metadata(self, tmp_path: Path) -> None:
        """Branch-B prep reads raw GWAS, applies QC, outputs with metadata."""
        import json

        gwas_path = tmp_path / "raw_gwas.tsv"
        _write_pgc_gwas(gwas_path, n=20)

        out_parquet = tmp_path / "gwas_spx.parquet"
        out_meta = tmp_path / "gwas_spx.meta.json"

        prepare_gwas_for_spredixcan(
            raw_gwas_path=gwas_path,
            output_parquet=out_parquet,
            output_meta=out_meta,
            genome_build="GRCh38",
            target_build="GRCh38",
        )

        result = pd.read_parquet(out_parquet)
        assert len(result) > 0
        assert "SNP" in result.columns

        with open(out_meta) as f:
            meta = json.load(f)
        assert meta["spredixcan_artifact"] is True
        assert meta["bim_harmonization_applied"] is False
        assert meta["branch_b_source"] == "raw_gwas"
        assert meta["genome_build"] == "GRCh38"

    def test_no_bim_harmonization_applied(self, tmp_path: Path) -> None:
        """Branch-B prep must NOT apply BIM harmonization."""
        import json

        gwas_path = tmp_path / "raw_gwas.tsv"
        _write_pgc_gwas(gwas_path, n=20)

        out_parquet = tmp_path / "gwas_spx.parquet"
        out_meta = tmp_path / "gwas_spx.meta.json"

        prepare_gwas_for_spredixcan(
            raw_gwas_path=gwas_path,
            output_parquet=out_parquet,
            output_meta=out_meta,
            genome_build="GRCh37",
            target_build="GRCh37",
        )

        result = pd.read_parquet(out_parquet)

        ref_parquet = tmp_path / "gwas_ref.parquet"
        ref_meta = tmp_path / "gwas_ref.meta.json"
        df_with_bim, _ = prepare_gwas(
            input_path=gwas_path,
            reference_bim=None,
            genome_build="GRCh37",
        )

        assert len(result) == len(df_with_bim)

        with open(out_meta) as f:
            meta = json.load(f)
        assert meta["bim_harmonization_applied"] is False

    def test_metadata_provenance_from_raw_input(self, tmp_path: Path) -> None:
        """Metadata should indicate raw GWAS provenance, not standardized."""
        import json

        gwas_path = tmp_path / "raw_gwas.tsv"
        _write_pgc_gwas(gwas_path, n=10)

        out_parquet = tmp_path / "gwas_spx.parquet"
        out_meta = tmp_path / "gwas_spx.meta.json"

        prepare_gwas_for_spredixcan(
            raw_gwas_path=gwas_path,
            output_parquet=out_parquet,
            output_meta=out_meta,
            genome_build="GRCh37",
            target_build="GRCh37",
        )

        with open(out_meta) as f:
            meta = json.load(f)

        assert meta["branch_b_source"] == "raw_gwas"
        assert meta["bim_harmonization_applied"] is False
        assert meta["spredixcan_artifact"] is True
        assert "n_variants_spredixcan" in meta
        assert meta["n_variants_spredixcan"] > 0


# ---------------------------------------------------------------------------
# case/control N + population prevalence threaded into metadata
# ---------------------------------------------------------------------------


class TestCaseControlMetadataPropagation:
    def test_config_supplied_n_takes_priority(self, tmp_path: Path) -> None:
        f = tmp_path / "cc.txt"
        _write_case_control_gwas(f)
        _df, meta = prepare_gwas(
            input_path=f,
            genome_build="GRCh37",
            trait_type="case_control",
            n_cases=1234,
            n_controls=5678,
            population_prevalence=0.01,
        )
        assert meta.n_cases == 1234
        assert meta.n_controls == 5678
        assert meta.n_total_cases_controls == 1234 + 5678
        assert meta.case_control_n_source == "config"
        assert meta.population_prevalence == 0.01

    def test_falls_back_to_column_median(self, tmp_path: Path) -> None:
        f = tmp_path / "cc.txt"
        _write_case_control_gwas(f)  # constant N_CAS=10000, N_CON=40000
        _df, meta = prepare_gwas(
            input_path=f,
            genome_build="GRCh37",
            trait_type="case_control",
        )
        assert meta.n_cases == 10000
        assert meta.n_controls == 40000
        assert meta.case_control_n_source == "gwas_column_median"
        assert meta.population_prevalence is None

    def test_quantitative_no_case_control_n(self, tmp_path: Path) -> None:
        f = tmp_path / "pgc.txt"
        _write_pgc_gwas(f)  # only N column
        _df, meta = prepare_gwas(input_path=f, genome_build="GRCh37")
        assert meta.n_cases is None
        assert meta.n_controls is None
        assert meta.case_control_n_source is None

    def test_metadata_json_roundtrip(self, tmp_path: Path) -> None:
        # New fields must survive model_dump -> JSON -> model_validate_json so the
        # MR sidecar reload path works.
        from repogen.data.schemas import GWASMetadata

        f = tmp_path / "cc.txt"
        _write_case_control_gwas(f)
        _df, meta = prepare_gwas(
            input_path=f, genome_build="GRCh37", trait_type="case_control",
            n_cases=100, n_controls=200, population_prevalence=0.02,
        )
        reloaded = GWASMetadata.model_validate_json(meta.model_dump_json())
        assert reloaded.n_cases == 100
        assert reloaded.n_controls == 200
        assert reloaded.population_prevalence == 0.02

    def test_archived_sidecar_without_new_fields_still_loads(self) -> None:
        # Backward compatibility: an old sidecar lacking the newer fields must
        # validate with None defaults.
        from repogen.data.schemas import GWASMetadata

        old = '{"genome_build": "GRCh37", "trait_type": "case_control"}'
        meta = GWASMetadata.model_validate_json(old)
        assert meta.n_cases is None
        assert meta.population_prevalence is None
