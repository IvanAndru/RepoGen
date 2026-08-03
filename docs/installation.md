# Installation

RepoGen needs a Linux environment. Native Linux, WSL2 on Windows, and HPC
login nodes are all supported and tested. macOS is not supported, because
several external tools (MAGMA, PLINK) are distributed as Linux binaries.

## 1. Choose an environment method

### Conda (recommended)

```bash
git clone https://github.com/IvanAndru/RepoGen.git
cd RepoGen
conda env create -f envs/repogen.yaml
conda activate repogen
pip install -e .
```

`envs/repogen.yaml` gives version ranges, so the solver picks current
compatible releases. Use this if you want a working environment.

### Pinned lock file (for reproducibility)

```bash
conda create -n repogen --file envs/repogen.linux-64.lock
conda activate repogen
pip install -r envs/repogen-pip.linux-64.txt
pip install -e .
```

The lock file records exact versions and build hashes for every package. Use
this when you need to reproduce a published result, or when a solver upgrade
changes numerical output. As the filenames say, these locks are for linux-64
and will not resolve on another platform.

Note that the pip requirements file deliberately does not pin `repogen`
itself; the package is always installed from the repository you have checked
out.

### Container

A container image is built and published by CI on tagged releases:

```bash
apptainer pull docker://ghcr.io/ivanandru/repogen:v1.0.0
apptainer exec repogen_v1.0.0.sif repogen info
```

The image contains the Python environment and PLINK, but **not MAGMA**, whose
licence forbids redistribution. Fetch MAGMA separately with
`repogen setup-resources` and bind-mount the resource directory into the
container.

Building the image yourself requires root, which HPC systems do not grant.
Build with Docker on a machine you control, or use the CI-published image.

## 2. Verify

```bash
repogen info
```

This prints the RepoGen version, the Python interpreter in use, and the
versions of key dependencies and external tools. If an external tool is
missing, it says so rather than failing later mid-run.

## 3. External tools

Two tools are called as subprocesses rather than imported.

| Tool | Purpose | How it is obtained |
|---|---|---|
| MAGMA | Gene and gene-set analysis (Branch A) | `repogen setup-resources` downloads it to `<resource_dir>/bin/magma` |
| PLINK 1.9 | LD reference handling | conda (`envs/repogen.yaml`) |

S-PrediXcan is **not** an external dependency. RepoGen implements the
algorithm (Barbeira et al. 2018) natively and reads PredictDB model files
directly, so there is no MetaXcan installation to manage.

RepoGen locates MAGMA in this order: the path set in your config, then
`<resource_dir>/bin/magma`, then `PATH`. So an explicitly configured path
always wins, and a MAGMA installed by `setup-resources` is found without
touching your `PATH`.

### Why MAGMA is not in the conda environment

The MAGMA licence permits free academic use but forbids redistribution, so it
cannot be bundled in the conda environment or the container image. Note that
the `magma` package on conda-forge is an unrelated GPU linear-algebra library;
installing it will not give you the genetics tool. A test in the suite asserts
that this name stays out of the environment file.

## 4. Optional dependency groups

```bash
pip install -e ".[lincs]"   # Branch B: cmapPy, rapidfuzz, h5py for LINCS GCTX
pip install -e ".[dev]"     # pytest, flake8, mypy
pip install -e ".[all]"     # everything
```

Branch B cannot read the LINCS L1000 GCTX file without the `lincs` group. The
import is deferred and raises a clear message pointing here if the group is
missing, rather than failing at import time.

## 5. HPC notes

On a cluster, install into your home directory and keep data on scratch:

```bash
# Login node
conda env create -f envs/repogen.yaml
conda activate repogen
pip install -e .
```

Two things matter on a shared filesystem:

- **Run from a directory you own.** Snakemake takes a lock on the working
  directory, and only one driver can run there at a time.
- **Keep the merged config off node-local storage.** RepoGen writes it beneath
  your `output_dir`, not `/tmp`, because every submitted job re-reads it from a
  compute node and `/tmp` is not shared.

For cluster submission see the `--profile` section of the
[README](../README.md) and `profiles/create/config.yaml`.

## 6. Running the tests

```bash
pytest -q
```

The suite is self-contained and needs no external data or network access.
Tests that would require a real MAGMA binary or large downloads are skipped.
