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


## File Changes Summary

| File | Action |
|------|--------|
| `unite/continuum.py` | **NEW** - Continuum interface + implementations |
| `unite/validation.py` | **NEW** - Testing framework |
| `unite/model.py` | Integrate continuum interface, absorption RT, flexible tying |
| `unite/optimized.py` | Add `absorption_transmission()` |
| `unite/parameters.py` | Add flexible tying parser |
| `unite/priors.py` | Add temperature, tau, beta priors |
| `unite/defaults.py` | Add tau bounds, new linetypes |
| `unite/testing.py` | Refactor to use validation framework |

---

## Implementation Order (Test-Driven)

### Phase 1 - Testing Framework (GATE: Must be 100% functional before proceeding)

| Step | Task | Test | Commit |
|------|------|------|--------|
| 1.1 | Create `unite/validation.py` skeleton | Import works | `git commit -m "Add validation.py skeleton"` |
| 1.2 | Implement `SyntheticLine`, `SyntheticContinuum` dataclasses | Unit test dataclasses | `git commit -m "Add synthetic line/continuum dataclasses"` |
| 1.3 | Implement `inject_synthetic_lines()` (generalize from testing.py) | Inject 2 Gaussians, verify flux added | `git commit -m "Add generic synthetic line injection"` |
| 1.4 | Implement `ValidationSuite.fit()` | Run fit on synthetic, get samples | `git commit -m "Add ValidationSuite.fit()"` |
| 1.5 | Implement `ValidationSuite.validate()` with metrics | Verify flux recovery within 10% | `git commit -m "Add validation metrics"` |
| 1.6 | Implement `verify_csv_output()` | Generate CSV, compare to reference | `git commit -m "Add CSV output verification"` |
| 1.7 | Generate reference CSV baseline from current model | Save to `tests/reference/` | `git commit -m "Add reference CSV baseline"` |
| 1.8 | Full validation test: inject → fit → validate → CSV check | **ALL PASS** | `git commit -m "Phase 1 complete: testing framework functional"` |

**GATE CHECK**: All 1.x tests must pass before proceeding to Phase 2.

---

### Phase 2 - Continuum Foundation (Keep old model, add V2)

| Step | Task | Test | Commit |
|------|------|------|--------|
| 2.1 | Create `continuum.py` with `ContinuumModel` ABC | Import, instantiate LinearContinuum | `git commit -m "Add ContinuumModel interface"` |
| 2.2 | Extract `LinearContinuum` from optimized.py | Unit test: same output as linearContinua() | `git commit -m "Extract LinearContinuum class"` |
| 2.3 | Create `multiSpecModelV2` in model.py using ContinuumModel | Runs without error | `git commit -m "Add multiSpecModelV2 with continuum interface"` |
| 2.4 | A/B test: V2 produces identical output to original | CSV diff = 0 | `git commit -m "Verify V2 matches original model"` |
| 2.5 | Add `model_version='v1'` parameter to NIRSpecFit | Switch between models | `git commit -m "Add model version switching to NIRSpecFit"` |

**GATE CHECK**: V2 must produce identical CSV output to V1 before proceeding.

---

### Phase 3 - Continuum Models (V2 only)

| Step | Task | Test | Commit |
|------|------|------|--------|
| 3.1 | Implement `planck_function()` | Unit test: matches scipy reference | `git commit -m "Add Planck function"` |
| 3.2 | Implement `BlackbodyContinuum` | Inject BB continuum, recover T within 20% | `git commit -m "Add BlackbodyContinuum"` |
| 3.3 | Implement `ModifiedBlackbodyContinuum` | Inject MBB, recover T, beta within tolerance | `git commit -m "Add ModifiedBlackbodyContinuum"` |
| 3.4 | Implement `CompositeContinuum` | Linear + BB composite, recover both | `git commit -m "Add CompositeContinuum"` |
| 3.5 | Add continuum config schema parsing | JSON config loads correctly | `git commit -m "Add continuum config parsing"` |

---

### Phase 4 - Absorption RT (V2 only)

| Step | Task | Test | Commit |
|------|------|------|--------|
| 4.1 | Add `absorption_partial_covering()` to optimized.py | Unit test: correct transmission | `git commit -m "Add partial covering absorption function"` |
| 4.2 | Add tau/covering priors to defaults.py, priors.py | Priors sample correctly | `git commit -m "Add absorption tau/covering priors"` |
| 4.3 | Modify `multiSpecModelV2` for RT absorption | Model runs with absorption config | `git commit -m "Integrate RT absorption in V2"` |
| 4.4 | Inject absorption line, validate tau recovery | tau recovered within 30% | `git commit -m "Validate absorption recovery"` |

---

### Phase 5 - Flexible Tying (V2 only)

| Step | Task | Test | Commit |
|------|------|------|--------|
| 5.1 | Add `TyingGroup` dataclass to parameters.py | Unit test dataclass | `git commit -m "Add TyingGroup dataclass"` |
| 5.2 | Implement `parse_tying_config()` | Parse sample config correctly | `git commit -m "Add flexible tying parser"` |
| 5.3 | Update V2 to use flexible tying matrices | Model runs with new tying | `git commit -m "Integrate flexible tying in V2"` |
| 5.4 | Backward compat: legacy config produces same matrices | Matrix diff = 0 | `git commit -m "Verify backward compat for tying"` |
| 5.5 | Test: tie Ha/Hb z, leave OIII independent | Correct parameter sharing | `git commit -m "Validate flexible tying"` |

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
