# unite: Uniform NIRSpec Inference (Turbo) Engine

### By Raphael Erik Hviding

Fast/efficient Bayesian inference of emission lines from multiple NIRSpec spectra simultaneously using [NumPyro](https://num.pyro.ai/).

---

## Changes on `dev` (relative to `main`)

This branch extends unite with a modular continuum model system, multiple fitting modes, improved diagnostics, and a structured result container. The original V1 model is preserved as a baseline.

### V2 Continuum Model System

The V1 model uses a hardcoded piecewise linear continuum. V2 introduces a `ContinuumModel` abstract base class with pluggable continuum types, all evaluated in the rest frame:

| Type | Config | Description |
|------|--------|-------------|
| `linear` | `{}` (default) | Piecewise linear (tilt + offset per region) |
| `chebyshev` | `{'type': 'chebyshev', 'order': 2}` | Chebyshev polynomial per region |
| `bspline` | `{'type': 'bspline', 'n_knots': 5, 'degree': 3}` | B-spline with WLS-initialized coefficients |
| `bernstein` | `{'type': 'bernstein', 'degree': 4}` | Bernstein polynomial with NNLS-initialized coefficients |
| `blackbody` | `{'type': 'blackbody'}` | Planck function B_lambda(T) |
| `modified_blackbody` | `{'type': 'modified_blackbody', 'beta': 1.5}` | A * B_lambda(T) * (lambda/lambda_0)^beta |
| `attenuated_blackbody` | `{'type': 'attenuated_blackbody'}` | A * B_lambda(T) * exp(-tau * extinction) |
| `none` | `{'type': 'none'}` | Zero continuum (pre-subtracted data) |

Configure via the `"continuum"` key in the JSON config:

```json
{
    "Name": "my_fit",
    "Unit": "Angstrom",
    "continuum": {"type": "bspline", "n_knots": 5, "degree": 3},
    "Groups": { ... }
}
```

### Fitting Modes

Three fitting modes control which pixels enter the likelihood:

| Mode | Pixels used | Default for |
|------|------------|-------------|
| `lines` | Windows around emission lines | `linear`, `chebyshev` |
| `full` | Entire spectral range | `blackbody`, `bspline`, `bernstein` |
| `continuum` | All pixels except masked emission lines | Set via `"fitting_mode": "continuum"` |

Override in config:

```json
{"continuum": {"type": "linear"}, "fitting_mode": "continuum"}
```

### FitResult Dataclass

`NIRSpecFit()` returns a `FitResult` dataclass instead of a plain dict:

```python
results = NIRSpecFit(config, rows=spec_table, output_directory="out")

results.summary()       # formatted summary table
print(results)          # one-liner repr
results.samples         # attribute access
results['samples']      # backward-compatible dict access
```

`summary()` prints spectra, line parameters (z, flux, FWHM with uncertainties), continuum parameters, and fit diagnostics (WAIC, calibration).

### Resolution-Aware Line Masking

Line masking now accounts for instrumental resolution. At PRISM resolution (~R=100), the default velocity-based mask (1000 km/s) is narrower than a single pixel. The mask now uses `max(base_pad, min(LINEPAD, local_FWHM))` where `local_FWHM` is estimated from pixel spacing, ensuring adequate masking at all spectral resolutions.

### Flux Regularization

For degenerate emission/absorption line pairs (e.g., broad emission + narrow absorption at the same wavelength), a soft Gaussian prior can prevent runaway fluxes:

```python
results = NIRSpecFit(config, rows=spec_table, flux_reg=10.0)
```

The penalty is `factor('flux_reg', -0.5 * sum(flux_params^2) / flux_reg^2)` and only targets lines that share a rest-frame wavelength (degenerate pairs). Well-separated lines are unaffected.

### Plotting Improvements

- **Gap-break rendering**: NaN insertion at wavelength gaps prevents matplotlib from drawing lines across masked/disjoint regions
- **Rest-frame axis**: Top axis shows rest-frame wavelengths with tick marks (no duplicate labels)
- **Config-based fit region restriction**: Optional `"Region": [lo, hi]` in config to restrict fitting to a wavelength subrange

### Other Changes

- **LINEPAD** increased from 3500 to 5000 km/s for broader line coverage
- **Line masking refactored** to compute once per spectrum and reuse
- **WAIC** computed with numerically stable `logsumexp`
- **Dynamic calibration export**: CSV includes all spectra (not hardcoded to PRISM)
- **`orthax`/`splinex` removed**: Chebyshev evaluation uses internal Clenshaw recurrence; B-spline uses Cox-de Boor
- **Line types**: absorption, emission, narrow, broad, lorentzian, exponential, outflow
- **FeII/FeI/TiII** absorption lines added from Kurucz
- **Config defaults**: override `LINEPAD`, `LINEDETECT`, `CONTINUUM` from config
- **Tests** in `tests/`, examples in `examples/`

## Installation

```bash
pixi install        # install dependencies
pixi run python     # run with managed environment
```

## Quick Start

```python
from unite.fitting import NIRSpecFit
from unite.utils import download_spectra
import json

# Load config
with open("examples/cliff.json") as f:
    config = json.load(f)

# Download spectra
spec_names = ["prism.spec.fits", "g395m.spec.fits"]
spec_table = download_spectra(spec_names, spectra_directory="examples/spectra")

# Fit
results = NIRSpecFit(config, rows=spec_table, output_directory="out", N=500)
results.summary()
```

## License

GPLv3
