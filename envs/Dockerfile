# =============================================================================
# RepoGen container image
# =============================================================================
# Build (from the repository root, note the -f path):
#     docker build -f envs/Dockerfile -t repogen:dev .
#
# Run on an HPC system with Apptainer/Singularity, which converts the image
# automatically and runs it as your own user:
#     apptainer pull repogen.sif docker://ghcr.io/<owner>/repogen:<version>
#     apptainer exec --bind /scratch/prj/<proj>:/scratch/prj/<proj> \
#         repogen.sif repogen run --config configs/config.yaml --step branch_a
#
# WHAT IS NOT IN HERE: MAGMA. Its licence states the binaries may not be
# redistributed, so shipping it inside a published image is not permitted.
# `repogen setup-resources` downloads it from the official distribution into
# <resource_dir>/bin/magma, which the pipeline resolves at runtime. PLINK is
# GPL-licensed and is included via conda.
# -----------------------------------------------------------------------------
FROM condaforge/miniforge3:26.3.2-3

LABEL org.opencontainers.image.title="RepoGen" \
      org.opencontainers.image.description="Genomic-driven drug repurposing: MAGMA gene/pathway/drug enrichment, S-PrediXcan signature reversal, and cis-eQTL Mendelian randomisation" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/IvanAndru/RepoGen"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- Environment ---------------------------------------------------------
# Built from the lock file, not envs/repogen.yaml: the lock pins exact
# versions, so an image built today and one built in five years are identical.
COPY envs/repogen.linux-64.lock envs/repogen-pip.linux-64.txt /tmp/envs/

RUN conda create --name repogen --file /tmp/envs/repogen.linux-64.lock --yes \
    && conda run -n repogen python -m pip install --no-cache-dir --no-deps \
         -r /tmp/envs/repogen-pip.linux-64.txt \
    && conda clean --all --force-pkgs-dirs --yes \
    && rm -rf /tmp/envs

# Put the environment first on PATH rather than relying on `conda run`, so
# `docker exec`, `apptainer exec` and Snakemake subprocesses all resolve the
# same interpreter without an activation wrapper.
ENV PATH=/opt/conda/envs/repogen/bin:$PATH \
    CONDA_DEFAULT_ENV=repogen

# --- Application ---------------------------------------------------------
WORKDIR /opt/repogen

COPY pyproject.toml ./
COPY repogen ./repogen
COPY workflows ./workflows
COPY profiles ./profiles
COPY configs ./configs
COPY tests ./tests

# setuptools-scm derives the version from git metadata, which is absent in a
# build context. The build argument supplies it instead; CI passes the release
# tag so `repogen --version` inside the image matches the published tag.
ARG REPOGEN_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${REPOGEN_VERSION}

RUN python -m pip install --no-cache-dir --no-deps . \
    && python -c "import repogen; print('installed repogen', repogen.__version__)"

# Lets the CLI find the workflow when the working directory is the user's data
# directory rather than this one (the usual case under Apptainer).
ENV REPOGEN_WORKFLOWS=/opt/repogen/workflows

# Fail the build rather than ship an image whose console script is broken.
RUN repogen info

CMD ["repogen", "--help"]
