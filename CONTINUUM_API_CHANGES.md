# Continuum Model API Changes

## Summary

Unified and simplified the continuum model architecture for UNITE V2.

## Key Changes

### 1. **All Continuum Models Use REST-FRAME Wavelengths**
   - Linear, Blackbody, and Modified Blackbody all evaluate in rest-frame
   - model.py handles the redshift conversion: `wave_rest = wave / (1 + z)`
   - Simplifies model code - no more conditional frame logic

### 2. **Encapsulated Parameter Sampling**
   - Each continuum model implements: `sample_params(sample_fn, cont_regs_rest)`
   - LinearContinuum handles its own `plate(Nc)` internally
   - Blackbody/MBB sample scalar parameters directly
   - model.py has uniform interface for all models

### 3. **List-Based Compositing**
   - Removed `CompositeContinuum` class
   - model.py handles compositing by looping over `List[ContinuumModel]`
   - Config supports both single model and list of models
   - Example: `"Continuum": [{"Type": "linear"}, {"Type": "blackbody"}]`

### 4. **Simplified Validation API**

```python
# Easy continuum definition
cont = SyntheticContinuum.from_params(
    type='blackbody',          # or 'modified_blackbody', 'linear'
    Temperature=5000,          # Kelvin
    Normalization=1e-19,       # f_lambda at pivot (observed frame)
    Beta=1.5,                  # For modified blackbody only
    PivotMicron=1.0            # microns (rest-frame)
)

# Use in validation
suite = ValidationSuite(rows=spec_table, lines=lines, continuum=cont)
```

## Files Modified

### Core Architecture
- **`unite/continuum.py`**
  - Updated `ContinuumModel` ABC with `sample_params()` method
  - Removed `uses_regions` and `uses_restframe` properties
  - All models now implement `sample_params(sample_fn, cont_regs)`
  - Removed `CompositeContinuum` class
  - Updated `parse_continuum_config()` to return `List[ContinuumModel]`

- **`unite/model.py`**
  - `multiSpecModelV2` now accepts `List[ContinuumModel]`
  - Converts cont_regs to rest-frame once: `cont_regs_rest = cont_regs / (1 + z)`
  - Loops over models calling `sample_params()` uniformly
  - Sums continuum contributions: `continuum = sum(model.evaluate(...))`

- **`unite/fitting.py`**
  - Updated to use `continuum_models` (list) instead of single model
  - `parse_continuum_config()` now returns a list

### Validation
- **`unite/validation.py`**
  - Added `SyntheticContinuum.from_params()` for easy creation
  - Supports `type='blackbody'`, `'modified_blackbody'`, `'linear'`
  - `Normalization` parameter is f_lambda at pivot wavelength

### Tests
- **`tests/test_continuum.py`**
  - Updated all tests for new API
  - Test `sample_params()` method instead of old `get_priors()`
  - Test config parsing returns list

- **`tests/test_validation.py`**
  - All validation tests pass with new API

### Documentation
- **`examples/uniteplus_validation.ipynb`**
  - Updated with three continuum options:
    1. Default linear continuum (`cont = None`)
    2. Blackbody continuum (uncomment example)
    3. Modified blackbody continuum (uncomment example)
  - Shows proper `SyntheticContinuum.from_params()` usage

## Test Results

All 23 tests pass:
- 15 continuum unit tests
- 8 validation tests (including blackbody recovery)

## Migration Guide

### Old API (before)
```python
# Model.py had to check continuum type
if continuum_model.uses_regions:
    with plate(Nc):
        params = sample(...)
else:
    params = sample(...)

if continuum_model.uses_restframe:
    wave_rest = wave / (1 + z)
    continuum = model.evaluate(wave_rest, ...)
else:
    continuum = model.evaluate(wave, ...)
```

### New API (after)
```python
# Model.py just loops uniformly
cont_regs_rest = cont_regs / (1 + z)
wave_rest = wave / (1 + z)

for model in continuum_models:
    params = model.sample_params(sample, cont_regs_rest)
    continuum += model.evaluate(wave_rest, params, cont_regs_rest)
```

## Benefits

1. **Simpler model.py** - Uniform interface, no conditionals
2. **Cleaner separation** - Each model handles its own complexity
3. **Easier compositing** - Just pass a list of models
4. **Better validation UX** - Simple `.from_params()` API
5. **All rest-frame** - Consistent wavelength handling
