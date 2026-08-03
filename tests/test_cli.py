"""Tests for repogen CLI (``repogen run`` and helpers)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from click.testing import CliRunner

from repogen.cli import (
    _STEP_ALIASES,
    _build_snakemake_command,
    _resolve_step_target,
    _write_merged_config,
    main,
)


@pytest.fixture()
def minimal_config_file(tmp_path: Path) -> Path:
    """Write a minimal valid pipeline config YAML and a dummy GWAS file."""
    gwas = tmp_path / "gwas.tsv"
    gwas.write_text("SNP\tCHR\tBP\tP\nrs1\t1\t100\t0.01\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump({"study": {"name": "test_study", "gwas_input": str(gwas)}})
    )
    return cfg


FAKE_SNAKEFILE = Path("/fake/workflows/Snakefile")
FAKE_SNAKEMAKE_EXE = "/fake/bin/snakemake"


# --- 1. Dry-run command construction ------------------------------------

def test_dryrun_command_construction(minimal_config_file: Path) -> None:
    runner = CliRunner()
    captured_cmd: list[str] | None = None

    def mock_run(cmd: list[str], **kwargs: object) -> None:
        nonlocal captured_cmd
        captured_cmd = cmd

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch("repogen.cli.subprocess.run", side_effect=mock_run),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--dryrun", "--step", "all"],
        )

    assert result.exit_code == 0, result.output
    assert captured_cmd is not None
    assert "-n" in captured_cmd
    assert "all" in captured_cmd
    assert "-s" in captured_cmd


# --- 2. Alias mapping ---------------------------------------------------

@pytest.mark.parametrize(
    "alias,expected",
    [
        ("branch_a", "branch_a_complete"),
        ("magma", "magma_complete"),
        ("full", "full_report"),
        ("available", "report_available"),
        ("drug", "drug_complete"),
        ("correlation", "correlation_complete"),
        ("mr", "mr_complete"),
        ("branch_b", "branch_b_complete"),
        ("branch_c", "branch_c_complete"),
    ],
)
def test_step_alias_mapping(alias: str, expected: str) -> None:
    assert _resolve_step_target(alias) == expected


def test_alias_mapping_via_cli(minimal_config_file: Path) -> None:
    runner = CliRunner()
    captured_cmd: list[str] | None = None

    def mock_run(cmd: list[str], **kwargs: object) -> None:
        nonlocal captured_cmd
        captured_cmd = cmd

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch("repogen.cli.subprocess.run", side_effect=mock_run),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--dryrun", "--step", "branch_a"],
        )

    assert result.exit_code == 0, result.output
    assert captured_cmd is not None
    assert "branch_a_complete" in captured_cmd


# --- 3. Direct rule passthrough -----------------------------------------

def test_direct_rule_passthrough() -> None:
    assert _resolve_step_target("magma_gene") == "magma_gene"
    assert _resolve_step_target("prepare_gwas") == "prepare_gwas"


def test_direct_rule_passthrough_via_cli(minimal_config_file: Path) -> None:
    runner = CliRunner()
    captured_cmd: list[str] | None = None

    def mock_run(cmd: list[str], **kwargs: object) -> None:
        nonlocal captured_cmd
        captured_cmd = cmd

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch("repogen.cli.subprocess.run", side_effect=mock_run),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--dryrun", "--step", "magma_gene"],
        )

    assert result.exit_code == 0, result.output
    assert captured_cmd is not None
    assert "magma_gene" in captured_cmd


# --- 4. Merged temp config is used --------------------------------------

def test_merged_temp_config_used(minimal_config_file: Path) -> None:
    runner = CliRunner()
    captured_cmd: list[str] | None = None

    def mock_run(cmd: list[str], **kwargs: object) -> None:
        nonlocal captured_cmd
        captured_cmd = cmd

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch("repogen.cli.subprocess.run", side_effect=mock_run),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--dryrun", "--step", "all"],
        )

    assert result.exit_code == 0, result.output
    assert captured_cmd is not None

    cfg_idx = captured_cmd.index("--configfile") + 1
    used_path = captured_cmd[cfg_idx]
    assert used_path != str(minimal_config_file), "Should use merged config, not original"
    assert "merged_config" in used_path


def test_merged_config_contains_study_name(tmp_path: Path) -> None:
    """_write_merged_config round-trips correctly."""
    mock_cfg = MagicMock()
    mock_cfg.model_dump.return_value = {
        "study": {"name": "roundtrip_test"},
        "output_dir": str(tmp_path / "results"),
    }

    path = _write_merged_config(mock_cfg)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["study"]["name"] == "roundtrip_test"
    finally:
        path.unlink(missing_ok=True)


def test_merged_config_written_to_shared_storage_not_tmp(tmp_path: Path) -> None:
    """The merged config must not live in the system temp directory.

    Under a cluster executor each submitted job re-invokes Snakemake on a
    compute node and re-reads this file. /tmp is node-local, so a config
    written there exists only on the submitting host and every job dies with
    FileNotFoundError before running any work.
    """
    import tempfile as _tempfile

    output_dir = tmp_path / "results"
    mock_cfg = MagicMock()
    mock_cfg.model_dump.return_value = {
        "study": {"name": "cluster_test"},
        "output_dir": str(output_dir),
    }

    path = _write_merged_config(mock_cfg)
    try:
        # Must land beside the outputs, which are on shared storage by
        # definition. (A "not under the temp directory" assertion cannot be
        # used here: pytest's tmp_path itself lives under the system temp.)
        assert path.parent == output_dir / ".repogen", (
            f"merged config should sit under output_dir/.repogen, got {path}"
        )
        assert path.parent != Path(_tempfile.gettempdir()), (
            "merged config was written straight into the system temp directory"
        )
        assert path.is_file()
    finally:
        path.unlink(missing_ok=True)


def test_merged_config_name_is_process_specific(tmp_path: Path) -> None:
    """Concurrent runs sharing an output directory must not clobber each other."""
    import os as _os

    mock_cfg = MagicMock()
    mock_cfg.model_dump.return_value = {
        "study": {"name": "x"}, "output_dir": str(tmp_path / "results"),
    }
    path = _write_merged_config(mock_cfg)
    try:
        assert str(_os.getpid()) in path.name
    finally:
        path.unlink(missing_ok=True)


# --- 5. Non-zero Snakemake exit propagates ------------------------------

def test_nonzero_exit_propagates(minimal_config_file: Path) -> None:
    runner = CliRunner()

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch(
            "repogen.cli.subprocess.run",
            side_effect=subprocess.CalledProcessError(returncode=2, cmd=["snakemake"]),
        ),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--step", "all"],
        )

    assert result.exit_code == 2


# --- 6. Invalid cores rejected ------------------------------------------

def test_invalid_cores_rejected(minimal_config_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--config", str(minimal_config_file), "--cores", "0"],
    )
    assert result.exit_code != 0
    assert "Must be >= 1" in result.output or "Invalid" in result.output


# --- Extra: _build_snakemake_command unit tests -------------------------

def test_build_command_with_exe() -> None:
    with patch("repogen.cli.shutil.which", return_value="/usr/bin/snakemake"):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, False,
        )
    assert cmd[0] == "/usr/bin/snakemake"
    assert "-n" not in cmd
    assert "--cores" in cmd
    assert "4" in cmd


def test_build_command_dryrun_fallback() -> None:
    with (
        patch("repogen.cli.shutil.which", return_value=None),
        patch("importlib.util.find_spec", return_value=True),
    ):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "magma_complete", 2, True,
        )
    assert cmd[1] == "-m"
    assert cmd[2] == "snakemake"
    assert "-n" in cmd
    assert "magma_complete" in cmd


def test_no_snakemake_gives_friendly_error() -> None:
    """When snakemake is neither on PATH nor importable, raise ClickException."""
    with (
        patch("repogen.cli.shutil.which", return_value=None),
        patch("importlib.util.find_spec", return_value=None),
        pytest.raises(click.ClickException, match="not installed"),
    ):
        _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, False,
        )


def test_no_snakemake_friendly_error_via_cli(minimal_config_file: Path) -> None:
    """End-to-end: missing snakemake shows install guidance, not a traceback."""
    runner = CliRunner()
    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=None),
        patch("importlib.util.find_spec", return_value=None),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--step", "all"],
        )
    assert result.exit_code != 0
    assert "not installed" in result.output


def test_all_aliases_covered() -> None:
    """Ensure every documented alias exists."""
    expected = {
        "all", "full", "full_report", "available", "report_available",
        "magma", "drug", "correlation", "mr", "branch_a", "branch_b", "branch_c",
    }
    assert set(_STEP_ALIASES.keys()) == expected


# --- Resource constraints ------------------------------------------------

def test_branch_b_heavy_resource_always_present() -> None:
    """--resources branch_b_heavy=1 must always be in the Snakemake command."""
    with patch("repogen.cli.shutil.which", return_value="/usr/bin/snakemake"):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, False,
        )
    assert "--resources" in cmd
    res_idx = cmd.index("--resources")
    assert cmd[res_idx + 1] == "branch_b_heavy=1"


def test_branch_b_heavy_resource_via_cli(minimal_config_file: Path) -> None:
    """End-to-end: resource constraint appears in actual CLI invocation."""
    runner = CliRunner()
    captured_cmd: list[str] | None = None

    def mock_run(cmd: list[str], **kwargs: object) -> None:
        nonlocal captured_cmd
        captured_cmd = cmd

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch("repogen.cli.subprocess.run", side_effect=mock_run),
    ):
        result = runner.invoke(
            main, ["run", "--config", str(minimal_config_file), "--dryrun", "--step", "branch_b"],
        )

    assert result.exit_code == 0, result.output
    assert captured_cmd is not None
    assert "--resources" in captured_cmd
    res_idx = captured_cmd.index("--resources")
    assert captured_cmd[res_idx + 1] == "branch_b_heavy=1"


# --- setup-resources target directory ------------------------------------


class TestSetupResourcesTargetDir:
    """--target-dir must default to the configured resource root.

    A cluster run put resource_dir on scratch, and setup-resources still wrote
    into the working directory, so the derived LINCS gene list was generated
    somewhere nothing else looked for it.
    """

    def test_defaults_to_parent_of_resource_dir(self, tmp_path: Path) -> None:
        from repogen.cli import _default_target_dir
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "study:\n  name: T\n  gwas_input: g.tsv\n"
            "resource_dir: /scratch/proj/repogen/resources\n",
            encoding="utf-8",
        )
        assert _default_target_dir(cfg) == Path("/scratch/proj/repogen")

    def test_falls_back_to_cwd_without_a_config(self) -> None:
        from repogen.cli import _default_target_dir
        assert _default_target_dir(None) == Path(".")

    def test_falls_back_to_cwd_when_resource_dir_absent(self, tmp_path: Path) -> None:
        from repogen.cli import _default_target_dir
        cfg = tmp_path / "config.yaml"
        cfg.write_text("study:\n  name: T\n  gwas_input: g.tsv\n", encoding="utf-8")
        assert _default_target_dir(cfg) == Path(".")


# --- setup-resources --pipeline-config wiring ---------


class TestSetupResourcesCLI:
    """The primary ``repogen setup-resources`` CLI must expose
    ``--pipeline-config`` and forward it as a kwarg to the underlying
    function, matching the module CLI surface.

    Regression guard: an earlier patch added ``--pipeline-config`` only to
    the module CLI in ``repogen/data/resources.py``; the primary
    Click CLI did not expose it, so users following the docs hit
    ``Error: No such option: --pipeline-config``.  These tests lock the
    primary CLI surface in place.
    """

    def test_help_lists_pipeline_config_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["setup-resources", "--help"])
        assert result.exit_code == 0
        assert "--pipeline-config" in result.output

    def test_pipeline_config_forwarded_as_kwarg(self, tmp_path: Path) -> None:
        rcfg = tmp_path / "resources.yaml"
        rcfg.write_text("resources: {}\n")
        pcfg = tmp_path / "pipeline.yaml"
        pcfg.write_text(
            "drug_enrichment:\n  enable_expression_enrichment: true\n"
        )

        captured: dict = {}

        def fake_setup(*args: object, **kwargs: object) -> dict:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {}

        runner = CliRunner()
        with patch(
            "repogen.data.resources.setup_resources",
            side_effect=fake_setup,
        ):
            result = runner.invoke(
                main,
                [
                    "setup-resources",
                    "--config", str(rcfg),
                    "--target-dir", str(tmp_path),
                    "--pipeline-config", str(pcfg),
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"].get("pipeline_config") == pcfg
        assert captured["kwargs"].get("resources_config") == rcfg
        assert captured["kwargs"].get("target_dir") == tmp_path

    def test_pipeline_config_default_none(self, tmp_path: Path) -> None:
        rcfg = tmp_path / "resources.yaml"
        rcfg.write_text("resources: {}\n")

        captured: dict = {}

        def fake_setup(*args: object, **kwargs: object) -> dict:
            captured["kwargs"] = kwargs
            return {}

        runner = CliRunner()
        with patch(
            "repogen.data.resources.setup_resources",
            side_effect=fake_setup,
        ):
            result = runner.invoke(
                main,
                [
                    "setup-resources",
                    "--config", str(rcfg),
                    "--target-dir", str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"].get("pipeline_config") is None


# --- Cluster submission: --profile and Snakemake passthrough -------------

def test_profile_flag_absent_by_default() -> None:
    """Local runs must not acquire a profile they did not ask for."""
    with patch("repogen.cli.shutil.which", return_value="/usr/bin/snakemake"):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, False,
        )
    assert "--profile" not in cmd


def test_profile_flag_forwarded() -> None:
    with patch("repogen.cli.shutil.which", return_value="/usr/bin/snakemake"):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, False,
            profile=Path("profiles/create"),
        )
    assert "--profile" in cmd
    assert cmd[cmd.index("--profile") + 1] == str(Path("profiles/create"))


def test_extra_args_appended_verbatim() -> None:
    with patch("repogen.cli.shutil.which", return_value="/usr/bin/snakemake"):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, False,
            extra_args=["--rerun-triggers", "mtime"],
        )
    assert cmd[-2:] == ["--rerun-triggers", "mtime"]


def test_dryrun_flag_precedes_passthrough() -> None:
    """-n must not be swallowed as a value of a trailing passthrough option."""
    with patch("repogen.cli.shutil.which", return_value="/usr/bin/snakemake"):
        cmd = _build_snakemake_command(
            Path("/wf/Snakefile"), Path("/tmp/cfg.yaml"), "all", 4, True,
            extra_args=["--unlock"],
        )
    assert cmd.index("-n") < cmd.index("--unlock")


def test_profile_and_passthrough_via_cli(tmp_path: Path, minimal_config_file: Path) -> None:
    """End-to-end: both reach the Snakemake invocation."""
    profile_dir = tmp_path / "profiles" / "create"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("executor: slurm\n")

    runner = CliRunner()
    captured: list[str] | None = None

    def mock_run(cmd: list[str], **kwargs: object) -> None:
        nonlocal captured
        captured = cmd

    with (
        patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE),
        patch("repogen.cli.shutil.which", return_value=FAKE_SNAKEMAKE_EXE),
        patch("repogen.cli.subprocess.run", side_effect=mock_run),
    ):
        result = runner.invoke(
            main,
            [
                "run", "--config", str(minimal_config_file), "--dryrun",
                "--step", "branch_a", "--profile", str(profile_dir),
                "--", "--rerun-triggers", "mtime",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured is not None
    assert captured[captured.index("--profile") + 1] == str(profile_dir)
    assert captured[-2:] == ["--rerun-triggers", "mtime"]


def test_nonexistent_profile_rejected(minimal_config_file: Path) -> None:
    """A mistyped profile path must fail before any job is submitted."""
    runner = CliRunner()
    with patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE):
        result = runner.invoke(
            main,
            ["run", "--config", str(minimal_config_file),
             "--profile", "profiles/does_not_exist"],
        )
    assert result.exit_code != 0
    assert "does_not_exist" in result.output


def test_unknown_option_still_errors(minimal_config_file: Path) -> None:
    """Passthrough must not turn typos into silently forwarded arguments."""
    runner = CliRunner()
    with patch("repogen.cli._resolve_snakefile", return_value=FAKE_SNAKEFILE):
        result = runner.invoke(
            main,
            ["run", "--config", str(minimal_config_file), "--dryrunn"],
        )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


# --- Workflow discovery (container support) ------------------------------

def test_snakefile_found_in_working_directory(tmp_path: Path, monkeypatch) -> None:
    """A repository checkout wins, so local edits take effect."""
    wf = tmp_path / "workflows"
    wf.mkdir()
    (wf / "Snakefile").write_text("# workflow\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REPOGEN_WORKFLOWS", raising=False)

    from repogen.cli import _resolve_snakefile
    assert _resolve_snakefile() == wf / "Snakefile"


def test_snakefile_found_via_env_var(tmp_path: Path, monkeypatch) -> None:
    """Inside the container the working directory is the user's data folder."""
    installed = tmp_path / "opt" / "repogen" / "workflows"
    installed.mkdir(parents=True)
    (installed / "Snakefile").write_text("# workflow\n")

    workdir = tmp_path / "scratch" / "project"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("REPOGEN_WORKFLOWS", str(installed))

    from repogen.cli import _resolve_snakefile
    assert _resolve_snakefile() == installed / "Snakefile"


def test_working_directory_takes_precedence_over_env_var(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "checkout" / "workflows"
    local.mkdir(parents=True)
    (local / "Snakefile").write_text("# local\n")
    other = tmp_path / "installed" / "workflows"
    other.mkdir(parents=True)
    (other / "Snakefile").write_text("# installed\n")

    monkeypatch.chdir(tmp_path / "checkout")
    monkeypatch.setenv("REPOGEN_WORKFLOWS", str(other))

    from repogen.cli import _resolve_snakefile
    assert _resolve_snakefile() == local / "Snakefile"


def test_missing_snakefile_error_names_the_env_var(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REPOGEN_WORKFLOWS", raising=False)

    # Make every candidate miss, including the package-relative one that
    # legitimately resolves while the tests run inside a checkout.
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    from repogen.cli import _resolve_snakefile
    with pytest.raises(click.ClickException) as exc:
        _resolve_snakefile()
    assert "REPOGEN_WORKFLOWS" in str(exc.value)


# --- setup-resources --branch --------------------------------------------

def test_branch_option_listed_in_help() -> None:
    result = CliRunner().invoke(main, ["setup-resources", "--help"])
    assert result.exit_code == 0
    assert "--branch" in result.output


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--branch", "a"], {"a"}),
        (["--branch", "a", "--branch", "c"], {"a", "c"}),
        (["--branch", "a,c"], {"a", "c"}),
        (["--branch", "A"], {"a"}),
        ([], None),
    ],
)
def test_branch_selection_forwarded(tmp_path: Path, argv: list[str], expected) -> None:
    """Repeatable, comma-separated and mixed-case forms all normalise."""
    rcfg = tmp_path / "resources.yaml"
    rcfg.write_text("resources: {}\n", encoding="utf-8")
    captured: dict = {}

    def fake_setup(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {}

    with patch("repogen.data.resources.setup_resources", side_effect=fake_setup):
        result = CliRunner().invoke(
            main, ["setup-resources", "--config", str(rcfg), *argv],
        )

    assert result.exit_code == 0, result.output
    assert captured["branches"] == expected
