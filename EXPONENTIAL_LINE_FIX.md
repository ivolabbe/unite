# Exponential Line Type Fix

## Date
2026-02-04

## Summary
Fixed three critical bugs that prevented exponential line components from working. Exponential lines failed MCMC initialization with "Cannot find valid initial parameters" error.

## Bug #1: Grating/Filter Name Parsing

**File**: `unite/spectra.py` (lines 600-611)

**Problem**: Resolution and dispersion calibration files were looked up using the full grating name including filter (e.g., `g235m_f170lp`), but calibration files only include grating name (e.g., `g235m`).

**Symptom**: `FileNotFoundError` when loading spectra with grating names like `G235M_F170LP`.

**Fix**: Extract grating name by splitting on underscore before looking up calibration files.

```python
# Extract grating name (strip filter part if present, e.g., G235M_F170LP -> G235M)
grating = disperser.split('_')[0]
```

**Impact**: Critical - prevented ANY spectra from loading when grating column includes filter.

## Bug #2: LSF Formula Error

**File**: `unite/calibration.py` (line 90)

**Problem**: LSF function returned `scale / (λ/R) = scale*R/λ` instead of FWHM `scale*λ/R`.

**Symptom**:
- LSF FWHM values were ~800 μm instead of ~0.001 μm
- Caused NaN gradients in MCMC initialization for exponential lines
- Gradient computation failed because extremely large LSF values created numerical instability

**Fix**: Corrected formula to properly compute FWHM.

```python
# OLD (wrong):
return scale / (λ / jnp.polyval(coeffs, λ * conversion))

# NEW (correct):
return scale * λ / jnp.polyval(coeffs, λ * conversion)
```

**Impact**: Critical - prevented gradient computation, causing MCMC init failure for ALL line types (not just exponential), though only exposed by exponential due to bug #3.

## Bug #3: Non-differentiable sign() Functions

**File**: `unite/optimized.py` (lines 189-197, 224-237)

**Problem**: Laplace CDF and Gaussian-Laplace convolution integrand used `jnp.sign(t)` which is not differentiable at t=0, causing NaN gradients.

**Symptom**:
- MCMC initialization failed with "Cannot find valid initial parameters"
- Gradients for fwhm_orig, lsf_scale, and redshift_orig were NaN
- Only affected exponential line type (Laplace/exponential profiles)

**Fix**: Replaced `sign(t) * f(|t|)` with differentiable `jnp.where(t >= 0, f(t), -f(t))`.

**File 1**: `integrateLaplace()` in optimized.py
```python
# OLD (non-differentiable):
def laplace_cdf(t):
    return jnp.sign(t) * (1 - jnp.exp(-jnp.abs(t)))

# NEW (differentiable):
def laplace_cdf(t):
    return jnp.where(t >= 0, 1 - 0.5 * jnp.exp(-t), 0.5 * jnp.exp(t))
```

**File 2**: `_integrandGL()` in optimized.py
```python
# OLD (non-differentiable):
return jnp.sign(t) * jnp.exp(a * a) * (posterm - jnp.exp(-twota) * erfc(-t_abs + a))

# NEW (differentiable):
result_abs = jnp.exp(a * a) * (posterm - jnp.exp(-twota) * erfc(-t_abs + a))
return jnp.where(t >= 0, result_abs, -result_abs)
```

**Impact**: Critical - prevented exponential line type from working at all. NUTS requires finite gradients for Hamiltonian dynamics.

## Root Cause Analysis

The bugs interacted in a subtle way:
1. Bug #2 (LSF formula) existed for ALL line types but was masked by small LSF values in normal usage
2. Bug #3 (sign functions) only affected exponential line type
3. Bug #2 amplified Bug #3: wrong LSF values created extreme parameter combinations that exposed the sign() gradient issue
4. When user tried exponential lines, both bugs combined to cause NaN gradients and MCMC failure

## Testing

All three fixes verified:

✅ **Spectra loading**: G235M_F170LP grating now loads correctly
✅ **Gradients finite**: All parameter gradients (flux, fwhm, redshift, lsf_scale) are finite
✅ **Single exponential**: Fit succeeds with single exponential line
✅ **Dual components**: Fit succeeds with emission + exponential for same line (Hβ)

## Files Modified

1. `unite/spectra.py` - Grating name parsing
2. `unite/calibration.py` - LSF formula correction
3. `unite/optimized.py` - Differentiable Laplace CDF and convolution integrand

## Commits

1. Fix grating/filter name parsing for calibration file lookup
2. Fix LSF formula to return correct FWHM
3. Fix non-differentiable sign() functions in Laplace profiles
