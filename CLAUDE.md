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


## New Features
Always keep track of new features implemented, and TODO of new features in unite/TODO.md
