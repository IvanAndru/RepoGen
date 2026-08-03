# Golden Images for Publication-Ready Plots

These PNG files serve as reference images for pixel-diff regression tests.
Tests are gated behind `REPOGEN_GOLDEN=1` and skipped by default.

## Refreshing golden images

1. Set `REPOGEN_GOLDEN=1` in your environment.
2. Run the test suite: `pytest tests/test_plotting_pubready.py`
3. Manually eyeball each new PNG before committing.
4. Replace the files in this directory with the new outputs.

## When golden images diverge

If matplotlib version bumps cause small rendering differences, update
the golden images intentionally, never silently.
