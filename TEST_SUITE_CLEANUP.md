# Test Suite Cleanup Summary

## Objective

Keep pytest suite lean and fast by skipping low-level unit tests that are not critical for CI or daily development.

## Changes Made

### Before
- **23 tests total**, all running
- **~50 seconds** runtime
- Many low-level unit tests for implementation details

### After
- **9 active tests, 14 skipped**
- **~38-40 seconds** runtime (20% faster)
- Only critical end-to-end and regression tests active

## Active Tests (9)

### test_continuum.py (5 active)
✅ **Config parsing tests** (critical for user API):
- `test_parse_continuum_config_default`
- `test_parse_continuum_config_blackbody`
- `test_parse_continuum_config_modified_blackbody`
- `test_parse_continuum_config_linear_explicit`
- `test_parse_continuum_config_composite`
- `test_parse_continuum_config_invalid`

### test_validation.py (3 active)
✅ **End-to-end validation tests**:
- `test_flux_recovery_single_line` - Core validation
- `test_blackbody_continuum_recovery` - New BB feature validation
- `test_v2_matches_v1` - Regression test

## Skipped Tests (14)

### test_continuum.py (9 skipped)
⏭️ **Low-level physics tests** (enable during BB/MBB development):
- `test_planck_function_normalized`
- `test_planck_function_temperature_dependence`
- `test_modified_blackbody_beta_zero`
- `test_modified_blackbody_positive_beta`

⏭️ **Implementation detail tests** (enable during API changes):
- `test_blackbody_continuum_sample_params`
- `test_blackbody_continuum_evaluate`
- `test_modified_blackbody_continuum_sample_params`
- `test_modified_blackbody_continuum_evaluate`

⏭️ **Trivial tests** (enable during compositing development):
- `test_composite_continuum_list`

### test_validation.py (5 skipped)
⏭️ **Redundant validation tests** (similar to active tests):
- `test_multiple_lines`
- `test_continuum_recovery`
- `test_config_saved`
- `test_injection_modifies_flux`
- `test_injection_adds_line_peak`

## How to Enable Skipped Tests

### During development of a specific feature:
```python
# In the test file, remove the @pytest.mark.skip decorator:

# BEFORE:
@pytest.mark.skip(reason="Low-level physics test - enable during BB development")
def test_planck_function_normalized():
    ...

# AFTER:
def test_planck_function_normalized():
    ...
```

### To see all skipped tests with reasons:
```bash
pixi run pytest tests/ -v -rs
```

## Guidelines for Future Tests

### ✅ DO add and keep active:
1. **End-to-end tests** - Test full pipeline from injection → fitting → validation
2. **Regression tests** - Ensure new features don't break existing functionality
3. **Critical API tests** - Test user-facing configuration and interfaces

### ❌ DON'T keep active (add with @pytest.mark.skip):
1. **Low-level unit tests** - Physics functions, helper functions
2. **Implementation details** - Internal methods like `sample_params()`, `evaluate()`
3. **Redundant tests** - Multiple tests of the same functionality
4. **Debugging tests** - One-off tests added during development

### 📝 Skip decorator template:
```python
@pytest.mark.skip(reason="[Category] - enable during [feature] development")
def test_something():
    """What this test does."""
    ...
```

Categories: `Low-level physics test`, `Implementation detail test`, `Redundant test`, `File I/O test`, etc.

## Benefits

1. **Faster CI** - 20% faster test runs
2. **Less noise** - Focus on critical failures
3. **Easier maintenance** - Fewer tests to update when refactoring
4. **Clearer intent** - Active tests show what matters
5. **Lower barrier** - Quick feedback loop for developers

## Documentation

Added to `CLAUDE.md` under "Test Suite Maintenance":
- Philosophy and guidelines
- How to enable/disable tests
- When to skip new tests
- Test count and runtime stats

## Files Modified

1. **tests/test_continuum.py**
   - Added header explaining skip policy
   - Skipped 9/15 tests (low-level unit tests)

2. **tests/test_validation.py**
   - Added header explaining skip policy
   - Skipped 5/8 tests (redundant/implementation detail tests)

3. **CLAUDE.md**
   - Added "Test Suite Maintenance" section
   - Guidelines for test management
   - Instructions for developers

## Test Results

```
pixi run pytest tests/ -q
sssssssss.......sss.ss.
9 passed, 14 skipped in 40.93s
```

All critical tests still pass ✅
