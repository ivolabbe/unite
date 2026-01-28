# UNITE TODO

## Implemented

### Validation Framework (v0.2)
- [x] `SyntheticLine` dataclass for configuring synthetic lines
- [x] `ValidationResult` dataclass for test results
- [x] `ValidationSuite` class with:
  - [x] `inject()` - inject synthetic lines using real spectrum's LSF
  - [x] `generate_config()` - generate UNITE config from synthetic lines
  - [x] `fit()` - run the **actual NIRSpecFit pipeline** (not reimplementation)
  - [x] `validate()` - compare recovered vs injected parameters
  - [x] `plot_results()` - plot fitting results via plotResults()
- [x] `inject_synthetic_lines()` standalone function using actual LSF
- [x] JSON config saving for reproducibility
- [x] Updated existing notebook: `examples/uniteplus_validation.ipynb`
- [x] Pytest tests in `tests/test_validation.py` (5 tests, all passing)
- [x] Pretty ASCII result print via `result.pretty_print()`

### Key Design Decisions
- Uses **real spectra** as templates (loaded via `NIRSpecSpectra(rows)`)
- Uses the **actual LSF** from calibration files (`spec.lsf()`)
- Calls the **actual NIRSpecFit()** function (tests real pipeline)
- Saves config JSON for reproducibility

## TODO

### High Priority
- [x] Test validation framework with actual downloaded spectra (G235M tests pass)
- [ ] Add G395M/H validation test cases (higher resolution)

### Medium Priority
- [ ] Add absorption line test case
- [ ] Add exponential/lorentzian profile test cases
- [ ] CI/CD integration for validation tests

### Low Priority
- [ ] Support for multi-spectrum validation
- [ ] Reference baseline comparison for regression testing
