"""RepoGen: Genomic-driven drug repurposing pipeline."""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("repogen")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
