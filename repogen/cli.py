"""RepoGen command-line interface.

Registered as a console entry point (``repogen``) via ``pyproject.toml``.
Subcommands mirror the main pipeline operations:

* ``repogen run``            - execute the pipeline (via Snakemake)
* ``repogen validate``       - check configs and resources
* ``repogen setup-resources`` - download / verify external data
* ``repogen info``           - print version and environment details
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import click
import yaml

import repogen
from repogen.utils.logging import reconfigure_logging, setup_logging

_SETUP_ERRORS: tuple[type[Exception], ...] = (FileNotFoundError, ValueError, OSError)
try:
    import requests
    _SETUP_ERRORS = (*_SETUP_ERRORS, requests.RequestException)
except ImportError:
    pass

logger = setup_logging(__name__)

_STEP_ALIASES: dict[str, str] = {
    "all": "all",
    "full": "full_report",
    "full_report": "full_report",
    "available": "report_available",
    "report_available": "report_available",
    "magma": "magma_complete",
    "drug": "drug_complete",
    "correlation": "correlation_complete",
    "mr": "mr_complete",
    "branch_a": "branch_a_complete",
    "branch_b": "branch_b_complete",
    "branch_c": "branch_c_complete",
}


def _resolve_step_target(step: str) -> str:
    """Map a user-friendly step alias to a Snakemake target rule name."""
    return _STEP_ALIASES.get(step, step)


def _resolve_snakefile() -> Path:
    """Locate the Snakemake workflow entry point.

    Resolution order:
        1. ``workflows/`` beneath the working directory - a repository
           checkout, where local edits should win.
        2. ``$REPOGEN_WORKFLOWS`` - set by the container image, where the
           working directory is the user's data directory rather than the
           installed application.
        3. ``workflows/`` beside the installed package.
    """
    candidates = [Path.cwd() / "workflows" / "Snakefile"]

    env_dir = os.environ.get("REPOGEN_WORKFLOWS")
    if env_dir:
        candidates.append(Path(env_dir) / "Snakefile")

    candidates.append(Path(__file__).resolve().parents[1] / "workflows" / "Snakefile")

    for path in candidates:
        if path.is_file():
            return path
    raise click.ClickException(
        "Cannot find workflows/Snakefile. Run from the repository root, or set "
        "REPOGEN_WORKFLOWS to the directory containing it."
    )


def _write_merged_config(config_obj: object) -> Path:
    """Serialize a validated PipelineConfig to a file Snakemake can read.

    Written beneath the output directory rather than the system temp
    directory. Under a cluster executor, every submitted job re-invokes
    Snakemake on a compute node and re-reads this file; ``/tmp`` is
    node-local, so the job would fail with ``FileNotFoundError`` before
    running anything. The output directory is on shared storage by
    definition - it is where the results must land.

    The filename carries the process id so concurrent runs in one output
    directory cannot overwrite each other's config.
    """
    data = config_obj.model_dump(mode="json")  # type: ignore[union-attr]
    target_dir = Path(data.get("output_dir", "results")) / ".repogen"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"merged_config.{os.getpid()}.yaml"
    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _build_snakemake_command(
    snakefile: Path,
    configfile: Path,
    target: str,
    cores: int,
    dryrun: bool,
    profile: Optional[Path] = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Assemble the Snakemake subprocess command list.

    Args:
        snakefile: Workflow entry point.
        configfile: Merged, validated pipeline config.
        target: Snakemake target rule.
        cores: Local core limit. Under a cluster profile this bounds local
            rules only; the number of submitted jobs comes from the profile.
        dryrun: Plan without executing.
        profile: Directory holding a Snakemake profile, e.g. profiles/create
            for SLURM submission on CREATE.
        extra_args: Additional Snakemake arguments passed through verbatim.
    """
    import importlib.util

    exe = shutil.which("snakemake")
    if exe:
        cmd: list[str] = [exe]
    elif importlib.util.find_spec("snakemake") is not None:
        cmd = [sys.executable, "-m", "snakemake"]
    else:
        raise click.ClickException(
            "Snakemake is not installed. "
            "Install it (`pip install snakemake`) or activate an environment that includes it."
        )
    cmd += ["-s", str(snakefile), "--configfile", str(configfile), "--cores", str(cores), target]
    # Serialises the two memory-heavy Branch B rules against each other.
    cmd += ["--resources", "branch_b_heavy=1"]
    if profile is not None:
        cmd += ["--profile", str(profile)]
    if dryrun:
        cmd.append("-n")
    cmd += list(extra_args)
    return cmd


@click.group()
@click.version_option(version=repogen.__version__, prog_name="repogen")
def main() -> None:
    """RepoGen - genomic-driven drug repurposing pipeline."""


# --- repogen run --------------------------------------------------------
@main.command()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=True, path_type=Path),
    default="configs/config.yaml",
    show_default=True,
    help="Path to the pipeline config YAML.",
)
@click.option(
    "--reference", "reference_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Path to the reference data YAML. "
        "If omitted, auto-discovers reference.yaml next to the config file."
    ),
)
@click.option(
    "--step",
    type=str,
    default="all",
    show_default=True,
    help=(
        "Pipeline step/target to run. Aliases: all, full, available, magma, "
        "drug, correlation, mr, branch_a, branch_b, branch_c. "
        "Direct rule names (e.g. magma_gene) are also accepted."
    ),
)
@click.option("--cores", type=int, default=4, show_default=True, help="Max parallel cores.")
@click.option("--dryrun", is_flag=True, help="Show what would be run without executing.")
@click.option(
    "--profile", "profile",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Snakemake profile directory for cluster submission, "
        "e.g. profiles/create to run each rule as a SLURM job."
    ),
)
@click.argument("snakemake_args", nargs=-1, type=click.UNPROCESSED)
def run(
    config_path: Path,
    reference_path: Optional[Path],
    step: str,
    cores: int,
    dryrun: bool,
    profile: Optional[Path],
    snakemake_args: tuple[str, ...],
) -> None:
    """Run the full pipeline or a single step via Snakemake.

    Arguments after a ``--`` separator are forwarded to Snakemake unchanged::

        repogen run --step branch_a -- --rerun-triggers mtime
        repogen run --step branch_c --profile profiles/create -- --unlock
    """
    if cores < 1:
        raise click.BadParameter("Must be >= 1.", param_hint="'--cores'")

    from repogen.config.loader import load_config

    config = load_config(config_path, reference_path)
    reconfigure_logging(config.log_level)

    target = _resolve_step_target(step)
    snakefile = _resolve_snakefile()
    logger.info(
        "Starting pipeline: target=%s, cores=%d, dryrun=%s, profile=%s",
        target, cores, dryrun, profile or "none (local execution)",
    )

    merged_cfg = _write_merged_config(config)
    try:
        cmd = _build_snakemake_command(
            snakefile, merged_cfg, target, cores, dryrun,
            profile=profile, extra_args=snakemake_args,
        )
        logger.debug("Snakemake command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise click.ClickException(
                "Snakemake executable not found. "
                "Install it (`pip install snakemake`) or activate the correct environment."
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode)
    finally:
        try:
            merged_cfg.unlink()
        except OSError:
            pass


# --- repogen validate ---------------------------------------------------
@main.command()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=True, path_type=Path),
    default="configs/config.yaml",
    show_default=True,
)
@click.option(
    "--reference", "reference_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Path to the reference data YAML. "
        "If omitted, auto-discovers reference.yaml next to the config file."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero if any resource is missing, not just shared prerequisites.",
)
def validate(config_path: Path, reference_path: Optional[Path], strict: bool) -> None:
    """Validate configuration files and check required resources."""
    import pydantic
    import yaml

    from repogen.config.loader import load_config
    from repogen.config.validate import SHARED, check_resources, summarise_by_branch

    try:
        config = load_config(config_path, reference_path)
        click.secho(f"Configuration valid: study={config.study.name}", fg="green")
    except (FileNotFoundError, ValueError, pydantic.ValidationError, yaml.YAMLError) as exc:
        click.secho(f"Validation failed: {exc}", fg="red", err=True)
        raise SystemExit(1) from exc

    checks = check_resources(config)
    summary = summarise_by_branch(checks)

    current_branch = None
    for check in checks:
        if check.branch != current_branch:
            current_branch = check.branch
            n_ok, n_total = summary[check.branch]
            click.echo(f"\n  {check.branch}  [{n_ok}/{n_total}]")
        mark, colour = ("ok", "green") if check.ok else ("MISSING", "red")
        click.echo("    ", nl=False)
        click.secho(f"{mark:>7}", fg=colour, nl=False)
        suffix = f"  - {check.detail}" if (check.detail and not check.ok) else ""
        click.echo(f"  {check.name}: {check.location}{suffix}")

    click.echo()
    runnable = [
        branch for branch, (n_ok, n_total) in summary.items()
        if branch != SHARED and n_ok == n_total
    ]
    shared_ok = summary.get(SHARED, (0, 0))[0] == summary.get(SHARED, (0, 0))[1]

    if not shared_ok:
        click.secho(
            "Shared prerequisites are missing - no branch can run.", fg="red", err=True
        )
        raise SystemExit(1)

    if runnable:
        click.secho(f"Ready to run: {', '.join(runnable)}", fg="green")
    else:
        click.secho("No branch has all of its resources.", fg="yellow")

    missing = [c for c in checks if not c.ok]
    if missing:
        click.secho(
            f"{len(missing)} resource(s) missing; run `repogen setup-resources` "
            "for anything auto-downloadable.",
            fg="yellow",
        )
        if strict:
            raise SystemExit(1)


def _default_target_dir(pipeline_config: Path | None) -> Path:
    """Where downloads should land when --target-dir was not given.

    ``local_path`` entries in resources.yaml already carry a ``resources/``
    prefix, so the base directory is the parent of ``resource_dir``. Deriving
    it here means a config pointing at cluster scratch is honoured; otherwise
    setup-resources would write into the working directory while every other
    command read from the configured root, and the two would never meet.
    """
    if pipeline_config is None:
        return Path(".")
    try:
        raw = yaml.safe_load(pipeline_config.read_text(encoding="utf-8")) or {}
        resource_dir = raw.get("resource_dir")
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read resource_dir from %s: %s", pipeline_config, exc)
        return Path(".")
    if not resource_dir:
        return Path(".")
    base = Path(resource_dir).parent
    logger.info("Resource base directory taken from %s: %s", pipeline_config, base)
    return base


# --- repogen setup-resources --------------------------------------------
@main.command("setup-resources")
@click.option(
    "--config", "resources_path",
    type=click.Path(exists=True, path_type=Path),
    default="configs/resources.yaml",
    show_default=True,
    help="Path to the resources YAML.",
)
@click.option(
    "--target-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Base directory for downloaded resources. Defaults to the parent of "
        "'resource_dir' from --pipeline-config when that is given, and to the "
        "working directory otherwise."
    ),
)
@click.option(
    "--pipeline-config", "pipeline_config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Optional pipeline config.yaml path used to evaluate the "
        "'enabled_by:' gates on optional resources. "
        "When omitted, every gated resource skips with status "
        "'gated_skipped' (default-deny). Resources without an "
        "'enabled_by:' field are unaffected."
    ),
)
@click.option(
    "--branch",
    multiple=True,
    help=(
        "Fetch only what these branches need: a (MAGMA/drug/ATC), "
        "b (S-PrediXcan/signature reversal), c (Mendelian randomisation). "
        "Repeatable or comma-separated. Shared inputs - the reference panel, "
        "gene-ID dictionaries and drug-target databases - are always included "
        "because all three branches consume them. Omit to fetch everything."
    ),
)
def setup_resources(
    resources_path: Path,
    target_dir: Path | None,
    pipeline_config: Path | None,
    branch: tuple[str, ...],
) -> None:
    """Download and verify external data dependencies."""
    from repogen.data.resources import setup_resources as _setup_resources

    if target_dir is None:
        target_dir = _default_target_dir(pipeline_config)

    # Accept both --branch a --branch b and --branch a,b
    selected: set[str] | None = None
    if branch:
        selected = {
            part.strip().lower()
            for value in branch
            for part in value.split(",")
            if part.strip()
        }

    try:
        _setup_resources(
            resources_config=resources_path,
            target_dir=target_dir,
            pipeline_config=pipeline_config,
            branches=selected,
        )
    except _SETUP_ERRORS as e:
        logger.error("Resource setup failed: %s", e)
        raise click.Abort()


# --- repogen info -------------------------------------------------------
@main.command()
def info() -> None:
    """Print version, Python environment, and key dependency versions."""
    import platform
    import sys
    from importlib.metadata import version as pkg_version, PackageNotFoundError

    click.echo(f"repogen {repogen.__version__}")
    click.echo(f"Python  {sys.version}")
    click.echo(f"OS      {platform.system()} {platform.release()}")
    click.echo()

    deps = [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "statsmodels", "pydantic", "pyyaml", "click", "pyarrow",
        "snakemake", "mygene", "rich", "cmapPy", "pyliftover",
    ]
    for name in deps:
        try:
            ver = pkg_version(name)
            click.echo(f"  {name:20s} {ver}")
        except PackageNotFoundError:
            click.secho(f"  {name:20s} NOT INSTALLED", fg="yellow")


if __name__ == "__main__":
    main()
