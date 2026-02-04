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

**CRITICAL: Gradient Overflow with Physical Constants**

When incorporating new physics-based models, always check for gradient overflow issues in JAX autodiff:

- **Problem:** Computing `x = (h * c) / (λ * kb * T)` directly causes gradient overflow
  - JAX autodiff creates intermediate values during backprop that overflow to ±inf
  - This breaks MCMC initialization even when forward evaluation is finite

- **Solution:** Pre-compute constants before division
  ```python
  # BAD - gradient will overflow:
  x = (_H * _C_SI) / (wavelength_m * _KB * temp)

  # GOOD - pre-compute constant first:
  const = (_H * _C_SI) / (wavelength_m * _KB)
  x = const / temp
  ```

- **Testing:** Always verify gradients when adding new models:
  ```python
  grad_fn = jax.grad(lambda t: model_function(wave, t))
  assert jnp.isfinite(grad_fn(test_temp))
  ```

- **Example:** See `planck_function()` in `unite/optimized.py` (lines 85-115) for the pattern used in blackbody continuum

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

## File Operations

**CRITICAL: Never overwrite existing files unless explicitly instructed**

- **Always read a file first** before using Write or creating a new file with the same name
- When adding new functions to existing modules (e.g., `utils.py`), use Edit or Read+Edit, never Write
- If a file already exists, assume it contains important code that must be preserved
- Only create new files when:
  1. The user explicitly asks for a new file
  2. You've verified the file doesn't exist
  3. The filename is clearly for a new purpose (e.g., test scripts, examples)

**Example of what NOT to do:**
```python
# BAD - overwrites existing utils.py with critical functions!
Write("unite/utils.py", "def new_function(): ...")
```

**Correct approach:**
```python
# GOOD - read first, then append or edit
Read("unite/utils.py")  # Check what's there
Edit("unite/utils.py", old_string="...", new_string="... + new_function")
```

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
1. **Always** update the pytest tests in `tests/test_validation.py`
2. **Always** update the example notebook `examples/uniteplus_validation.ipynb` to match
3. **CRITICAL: Always run the notebook with Jupyter to test for errors** - don't just read the code
4. The notebook demonstrates the API for users; pytest tests are the source of truth for correctness
5. Keep the notebook's ValidationSuite usage consistent with the current API (e.g., SyntheticContinuum, parameter names)

**Notebook verification checklist:**
- Update imports to match new API
- Update function calls with correct parameters
- **MANDATORY: Execute the notebook to verify no errors before claiming it works:**
  ```bash
  # Create test script that executes notebook cells
  pixi run python /path/to/notebook_test.py
  ```
- Verify `result.pretty_print()` displays correctly
- Check that plots render without issues
- **CRITICAL: Never assume code works without testing** - isolating functions may not replicate the example notebook exactly
- If you modify validation.py or plotting.py, you MUST run the notebook to catch errors before committing changes

**Validation output:** Use `result.pretty_print()` to display a formatted table showing:
- Injected vs recovered values
- Uncertainty intervals (16th/84th percentiles)
- SNR for each line
- Pass/fail status per parameter

## Test Suite Maintenance

**Philosophy: Keep tests lean and fast. Skip low-level unit tests unless actively developing that feature.**

Current test status: **9 active tests, 14 skipped** (~38s runtime)

**Active tests (always run):**
- End-to-end validation tests (flux/BB recovery)
- Regression tests (V2 vs V1)
- Critical config parsing tests

**Skipped tests (enable during development):**
- Low-level physics tests (Planck function, etc.)
- Implementation detail tests (sample_params, evaluate)
- Redundant tests (multiple lines, injection details)

**To enable skipped tests during development:**
```python
# Remove the decorator from the specific test:
# @pytest.mark.skip(reason="...")  # Remove this line
def test_something():
    ...
```

**Guidelines for adding new tests:**
1. **Prefer end-to-end tests** over unit tests
2. **Skip unit tests by default** - add `@pytest.mark.skip(reason="...")`
3. **Only keep active:**
   - End-to-end validation tests
   - Regression tests
   - Tests for features under active development
4. **When feature is stable**, skip the detailed unit tests
5. **Document skip reason** clearly for future developers

**Running all tests (including skipped):**
```bash
pixi run pytest tests/ -v  # Shows skipped tests
pixi run pytest tests/ -v -rs  # Shows skip reasons
```

**DO NOT** let the test suite balloon with low-level unit tests over time. They slow down CI and make development tedious. Trust end-to-end tests to catch regressions.

## New Features & TODO.md Format

**TODO.md structure:**
- Simplified plan + implementation status tables
- Status icons in **second column** after step number:
  - ✅ Complete and working
  - 🛑 Implemented but broken
  - ⚪ Not yet implemented
- **No commit column** - keep tables clean and readable
- Each phase has a GATE CHECK describing completion criteria

**Updating TODO.md:**
- Add new features and planned changes as needed
- Mark status with appropriate icon (✅/🛑/⚪)
- Keep 'Implemented' section at top for completed major features
- Rank future work by priority (High/Medium/Low)
