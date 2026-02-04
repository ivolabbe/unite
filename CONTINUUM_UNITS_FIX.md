# Continuum Normalization Units Fix

## Issues Fixed

### 1. **Normalization Units** ✅
   - **Problem**: Normalization was treated as raw flux density, not in JWST standard units
   - **Expected**: Normalization in units of `1e-20 erg/s/cm²/Å` (JWST standard)
   - **Fix**: Scale by `1e-20` when converting to amplitude

### 2. **Config Generation for Non-Linear Continuum** ✅
   - **Problem**: `generate_config()` always created linear continuum config
   - **Expected**: Config should match the injected continuum type (BB/MBB)
   - **Fix**: Add "Continuum" section to config based on `self.continuum.continuum_type`

## Implementation

### SyntheticContinuum.from_params()

```python
# BEFORE: Normalization was raw flux density
bb_amplitude=Normalization  # Wrong!

# AFTER: Normalization in units of 1e-20
amplitude = Normalization * 1e-20  # Correct!
bb_amplitude=amplitude
```

### ValidationSuite.generate_config()

```python
# NEW: Add continuum config section
if self.continuum is not None and self.continuum.continuum_type != 'linear':
    if self.continuum.continuum_type == 'blackbody':
        config['Continuum'] = {
            'Type': 'blackbody',
            'PivotMicron': self.continuum.pivot_micron,
        }
    elif self.continuum.continuum_type == 'modified_blackbody':
        config['Continuum'] = {
            'Type': 'modified_blackbody',
            'PivotMicron': self.continuum.pivot_micron,
        }
```

## Usage Examples

### Blackbody Continuum
```python
# Normalization = 10 means f_lambda = 10 * 1e-20 = 1e-19 erg/s/cm²/Å
cont = SyntheticContinuum.from_params(
    type='blackbody',
    Temperature=5000,           # Kelvin
    Normalization=10,           # in units of 1e-20 erg/s/cm²/Å
    PivotMicron=0.65            # microns (rest-frame)
)

# At pivot wavelength 0.65 microns, continuum level = 10 (in plot units)
# Actual flux density = 10 * 1e-20 = 1e-19 erg/s/cm²/Å
```

### Modified Blackbody Continuum
```python
# Same units apply
cont = SyntheticContinuum.from_params(
    type='modified_blackbody',
    Temperature=5000,           # Kelvin
    Normalization=10,           # in units of 1e-20 erg/s/cm²/Å
    Beta=1.5,                   # Emissivity index
    PivotMicron=1.0             # microns (rest-frame)
)
```

## Validation

### Unit Tests
```python
# Test 1: Blackbody normalization
cont = SyntheticContinuum.from_params(type='blackbody', Temperature=5000, Normalization=10)
assert cont.bb_amplitude == 1e-19  # ✅ 10 * 1e-20

# Test 2: Modified blackbody normalization
cont = SyntheticContinuum.from_params(type='modified_blackbody', Temperature=5000, Normalization=1, Beta=1.5)
assert cont.bb_amplitude == 1e-20  # ✅ 1 * 1e-20

# Test 3: Linear continuum normalization
cont = SyntheticContinuum.from_params(type='linear', regions=..., angles=..., offsets=[5.0])
assert cont.offsets[0] == 5e-20  # ✅ 5 * 1e-20
```

### End-to-End Test
```python
# Create MBB continuum with Normalization=10
cont = SyntheticContinuum.from_params(
    type='modified_blackbody',
    Temperature=5000,
    Normalization=10,  # 10 * 1e-20 erg/s/cm²/Å
    Beta=1.5,
    PivotMicron=0.65
)

# Inject and fit
suite = ValidationSuite(rows=spec_table, lines=lines, continuum=cont)
suite.generate_config()  # Now includes "Continuum": {"Type": "modified_blackbody", ...}
suite.inject()
suite.fit()

# Validation shows MBB parameters recovered (not linear!)
result = suite.validate()
# Result shows bb_amplitude, bb_temperature, bb_beta recovery
```

## Files Modified

1. **`unite/validation.py`**
   - `SyntheticContinuum.from_params()`: Scale Normalization by 1e-20
   - `ValidationSuite.generate_config()`: Add continuum config section

2. **`examples/uniteplus_validation.ipynb`**
   - Updated documentation to clarify units
   - Changed comments from "erg/s/cm²/Å at pivot" to "in units of 1e-20 erg/s/cm²/Å"

## Test Results

All 8 validation tests pass ✅

## Breaking Change

**Users must update their code** if they were using explicit normalization values:

```python
# BEFORE (wrong interpretation):
Normalization=1e-19  # This was treated as raw flux density

# AFTER (correct):
Normalization=10     # This is in units of 1e-20, giving 10 * 1e-20 = 1e-19
```

The new interpretation matches JWST standard flux units where spectra are typically displayed in units of `1e-20 erg/s/cm²/Å`.
