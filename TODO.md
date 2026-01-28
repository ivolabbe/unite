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

## Implementation Order (Test-Driven)

### Phase 1 - Testing Framework (GATE: Must be 100% functional before proceeding)

| Step | Pass | Task | Test | Commit |
|------|------|------|------|--------|
| 1.1 | X | Create `unite/validation.py` skeleton | Import works | `git commit -m "Add validation.py skeleton"` |
| 1.2 | X | Implement `SyntheticLine` dataclass | Unit test dataclass fields | `git commit -m "Add SyntheticLine dataclass"` |
| 1.3 | X | Implement `ValidationResult` dataclass | Unit test result fields | `git commit -m "Add ValidationResult dataclass"` |
| 1.4 | X | Implement `inject_synthetic_lines()` | `test_injection_modifies_flux` | `git commit -m "Add inject_synthetic_lines function"` |
| 1.5 | X | Verify line peak injection | `test_injection_adds_line_peak` | `git commit -m "Verify line injection creates peak"` |

### Phase 2 - ValidationSuite Core (GATE: Single line recovery works)

| Step | Pass | Task | Test | Commit |
|------|------|------|------|--------|
| 2.1 | X | Implement `ValidationSuite.__init__()` | Loads spectra | `git commit -m "Add ValidationSuite init"` |
| 2.2 | X | Implement `ValidationSuite.inject()` | Modifies spectrum | `git commit -m "Add ValidationSuite.inject()"` |
| 2.3 | X | Implement `ValidationSuite.generate_config()` | Config has correct structure | `git commit -m "Add ValidationSuite.generate_config()"` |
| 2.4 | X | Implement `ValidationSuite.fit()` | Calls NIRSpecFit, saves results | `git commit -m "Add ValidationSuite.fit()"` |
| 2.5 | X | Implement `ValidationSuite.validate()` | `test_flux_recovery_single_line` | `git commit -m "Add ValidationSuite.validate()"` |

### Phase 3 - Extended Validation (GATE: Multi-line recovery works)

| Step | Pass | Task | Test | Commit |
|------|------|------|------|--------|
| 3.1 | X | Support multiple lines in config | `test_multiple_lines` | `git commit -m "Support multiple lines"` |
| 3.2 | X | Add JSON config saving | `test_config_saved` | `git commit -m "Save validation config JSON"` |
| 3.3 | X | Implement `ValidationResult.pretty_print()` | Output is human-readable | `git commit -m "Add pretty_print() for results"` |
| 3.4 | X | Add sigma-based tolerance | Pass/fail uses Nσ | `git commit -m "Use sigma-based validation tolerance"` |
| 3.5 | X | Implement `ValidationSuite.plot_results()` | Plots generated | `git commit -m "Add plot_results()"` |

### Phase 4 - Documentation & CI (GATE: Ready for production)

| Step | Pass | Task | Test | Commit |
|------|------|------|------|--------|
| 4.1 | X | Update example notebook | Notebook runs end-to-end | `git commit -m "Update validation notebook"` |
| 4.2 | X | Update CLAUDE.md | Docs reflect implementation | `git commit -m "Document validation framework"` |
| 4.3 |   | Add CI/CD integration | Tests run in CI | `git commit -m "Add validation to CI"` |

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
