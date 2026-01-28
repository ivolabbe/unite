# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UNITE (Uniform NIRSpec Inference Turbo Engine) is a Python package for fast Bayesian inference of emission lines from multiple JWST NIRSpec spectra simultaneously. It uses JAX for GPU acceleration and Numpyro for probabilistic programming.

## Development Commands

```bash
pixi run format    # Format code with Ruff
pixi run build     # Build distribution (requires 'build' environment)
```

## Environment Setup

Uses Pixi for environment management:
- `default` - Standard JAX environment
- `mac` - macOS with jax-metal GPU support
- `build` - For building distributions

## Architecture

**Pipeline flow:** JSON config → `NIRSpecSpectra` → `NIRSpecFit` → MCMC sampling → `plotResults`

**Core modules in `unite/`:**
- `spectra.py` - `NIRSpecSpectra` and `NIRSpecSpectrum` classes for loading/managing spectral data
- `fitting.py` - `NIRSpecFit` orchestrates fitting; `NIRSpecModelArgs` prepares model inputs
- `model.py` - `multiSpecModel()` defines the Numpyro probabilistic model with hierarchical structure
- `parameters.py` - Converts JSON config to sparse matrices (BCOO) for efficient parameter mapping
- `optimized.py` - JAX-compiled line profile integrals (Gaussian, Lorentzian, Exponential)
- `plotting.py` - `plotResults`, `plotRegionSingle` for visualization
- `defaults.py` - Prior ranges, line types, global constants
- `priors.py` - Bayesian prior distributions
- `calibration.py` - LSF and pixel offset calibrations

**Configuration:** JSON files define emission line groups with species, wavelengths, and line types (emission, absorption, broad, narrow, outflow, lorentzian, exponential).

**Key dependency:** `dotfit` provides `EmissionLines()` for building line configurations via `el.to_unite(groups)`.

### JAX/NumPyro Conventions
- Use `@jax.jit` for hot paths (see `broken_modified_bb.py`)
- NumPyro `sample()` calls must be inside model functions
- Sparse matrices via `jax.experimental.sparse.BCOO` for parameter tying
- Clip numerical values to prevent NaN gradients (see `_safe_log_expm1`)

## Configuration Format
Line configs use JSON (see `unite/examples/*.json`):
```json
{
  "Groups": {
    "broad1": {
      "TieRedshift": true,
      "TieDispersion": true,
      "Species": [{"Name": "Ha", "LineType": "broad", "Lines": [...]}]
    }
  }
}
```
`LineType` values: `narrow`, `broad`, `lorentzian`, `exponential`, `absorption`, `emission`, `outflow` (see `unite/defaults.py`)

### Data Flow
1. Config JSON defines line groups with tied redshifts/dispersions → `parameters.configToMatrices()`
2. FITS spectra loaded via `NIRSpecSpectra(rows)` → continuum regions computed
3. NumPyro model samples: flux, redshift offsets, FWHMs per group
4. Results saved to FITS tables + diagnostic plots

## Notebook Workflow
Example notebooks in `unite/examples/`


## Code Style

- Ruff formatter with single quotes
- Skip magic trailing comma


## Validation Framework

The validation framework (`unite/validation.py`) tests the **actual pipeline** end-to-end:
- Uses real spectra as templates (not synthetic/mocked spectra)
- Injects lines using the actual LSF from calibration files (`spec.lsf()`)
- Calls the real `NIRSpecFit()` function
- Saves config JSON for reproducibility

**Running validation tests:**
```bash
pixi run pytest tests/test_validation.py -v
```

**Important:** When making interface changes to `validation.py`:
1. Always update the pytest tests in `tests/test_validation.py`
2. Update the example notebook `examples/uniteplus_validation.ipynb` accordingly
3. The notebook is for manual inspection; pytest tests are the source of truth

**Validation output:** Use `result.pretty_print()` to display a formatted table showing:
- Injected vs recovered values
- Uncertainty intervals (16th/84th percentiles)
- SNR for each line
- Pass/fail status per parameter

## New Features
- keep a TODO of new features in unite/TODO.md, add new features, planned changes to the TODO list as needed, and rank features by priority and type
- always keep track of new features implemented, by ticking them off in the TODO (e.g. keep a 'implemented' and 'todo' section
