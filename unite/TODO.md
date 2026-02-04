# UNITE TODO

## Implemented

### Validation Framework (v0.2)
- [x] `SyntheticLine` dataclass for configuring synthetic lines
- [x] `SyntheticContinuum` dataclass with `from_params()` for BB/MBB injection
- [x] `ValidationResult` dataclass for test results
- [x] `ValidationSuite` class with:
  - [x] `inject()` - inject synthetic lines using real spectrum's LSF
  - [x] `generate_config()` - generate UNITE config from synthetic lines
  - [x] `fit()` - run the **actual NIRSpecFit pipeline** (not reimplementation)
  - [x] `validate()` - compare recovered vs injected parameters
  - [x] `plot_results()` - plot fitting results via plotResults()
  - [x] `plot_spectrum()` - plot full spectrum with fitted regions highlighted
- [x] `inject_synthetic_lines()` standalone function using actual LSF
- [x] Composite continuum support (multiple BB/MBB components)
- [x] Automatic line classification by FWHM (narrow < 700, broad > 1000 km/s)
- [x] JSON config saving for reproducibility
- [x] Updated existing notebook: `examples/uniteplus_validation.ipynb`
- [x] Pytest tests in `tests/test_validation.py` (6 tests, 3 passing, 3 skipped)
- [x] Pretty ASCII result print via `result.pretty_print()` with composite continuum display

### Key Design Decisions
- Uses **real spectra** as templates (loaded via `NIRSpecSpectra(rows)`)
- Uses the **actual LSF** from calibration files (`spec.lsf()`)
- Calls the **actual NIRSpecFit()** function (tests real pipeline)
- Saves config JSON for reproducibility

### Continuum Models (v0.3)
- [x] `ContinuumModel` ABC with `sample_params()` and `evaluate()` interface
- [x] `LinearContinuum` - piecewise linear continuum (default)
- [x] `BlackbodyContinuum` - pure blackbody (beta=0)
- [x] `ModifiedBlackbodyContinuum` - dust-like continuum with emissivity index
- [x] Composite continuum support (multiple MBB components with unique names)
- [x] `planck_function()` and `modified_blackbody()` in optimized.py
- [x] Temperature priors: hot (20k-100k), warm (2k-15k), dust (20-1500), default (1k-30k)
- [x] Beta prior for emissivity index (-2.0 to 2.0)
- [x] `parse_continuum_config()` for JSON config parsing
- [x] `plotFullSpectrum()` in plotting.py - visualize fitted vs unfitted regions
- [x] All continuum models operate in rest-frame wavelengths

---

## Implementation Roadmap

### Legend
- ✅ Complete and working
- 🛑 Implemented but broken
- ⚪ Not yet implemented

---

## File Changes Summary

| File | Action | Status |
|------|--------|--------|
| `unite/continuum.py` | **NEW** - Continuum interface + implementations (Linear, BB, MBB, Composite) | ✅ |
| `unite/validation.py` | **NEW** - Testing framework with composite continuum support | ✅ |
| `unite/optimized.py` | ADD `planck_function()`, `modified_blackbody()` | ✅ |
| `unite/priors.py` | ADD temperature_prior, beta_prior, amplitude_prior | ✅ |
| `unite/defaults.py` | ADD temperature bounds, beta bounds | ✅ |
| `unite/plotting.py` | ADD `plotFullSpectrum()` for fitted region visualization | ✅ |
| `unite/model.py` | Integrate continuum interface (V2), absorption RT, flexible tying | 🟡 Partial |
| `unite/parameters.py` | Add flexible tying parser | ⚪ |
| `tests/test_continuum.py` | **NEW** - Unit tests for continuum models | ⚪ |

---

### Phase 1 - Testing Framework

**GATE CHECK**: All 1.x tests must pass before proceeding to Phase 2.

| Step | Status | Task | Test |
|------|--------|------|------|
| 1.1 | ✅ | Create `unite/validation.py` skeleton | Import works |
| 1.2 | ✅ | Implement `SyntheticLine`, `SyntheticContinuum` dataclasses | Unit test dataclasses |
| 1.3 | ✅ | Implement `inject_synthetic_lines()` (generalize from testing.py) | Inject 2 Gaussians, verify flux added |
| 1.4 | ✅ | Implement `ValidationSuite.fit()` | Run fit on synthetic, get samples |
| 1.5 | ✅ | Implement `ValidationSuite.validate()` with metrics | Verify flux recovery within 10% |
| 1.6 | ✅ | Implement `verify_csv_output()` | Generate CSV, compare to reference |
| 1.7 | ✅ | Generate reference CSV baseline from current model | Save to `tests/reference/` |
| 1.8 | ✅ | Full validation test: inject → fit → validate → CSV check | **ALL PASS** |

---

### Phase 2 - Continuum Foundation

**GATE CHECK**: V2 must produce similar CSV output to V1 (test_v2_matches_v1 added). V2 is now default.

| Step | Status | Task | Test |
|------|--------|------|------|
| 2.1 | ✅ | Create `continuum.py` with `ContinuumModel` ABC | Import, instantiate LinearContinuum |
| 2.2 | ✅ | Extract `LinearContinuum` from optimized.py | Unit test: same output as linearContinua() |
| 2.3 | ✅ | Create `multiSpecModelV2` in model.py using ContinuumModel | Runs without error |
| 2.4 | ✅ | A/B test: V2 produces identical output to original | CSV diff within tolerance |
| 2.5 | ✅ | Add `model_version` parameter to NIRSpecFit, make V2 default | Switch between models |

---

### Phase 3 - Continuum Models (V2 only)

**GATE CHECK**: All continuum models implemented and tested. Composite continuum validation passes.

| Step | Status | Task | Test |
|------|--------|------|------|
| 3.1 | ✅ | Implement `planck_function()` in optimized.py | Numerical stability with safe_log_expm1 |
| 3.2 | ✅ | Implement `BlackbodyContinuum` | test_blackbody_continuum_recovery passes |
| 3.3 | ✅ | Implement `ModifiedBlackbodyContinuum` | Composite continuum test passes |
| 3.4 | ✅ | Implement composite continuum (multiple MBB) | test_composite_continuum_multi_line passes |
| 3.5 | ✅ | Add continuum config schema parsing | parse_continuum_config() handles composite configs |

---

### Phase 4 - Absorption RT (V2 only)

| Step | Status | Task | Test |
|------|--------|------|------|
| 4.1 | ⚪ | Add `absorption_partial_covering()` to optimized.py | Unit test: correct transmission |
| 4.2 | ⚪ | Add tau/covering priors to defaults.py, priors.py | Priors sample correctly |
| 4.3 | ⚪ | Modify `multiSpecModelV2` for RT absorption | Model runs with absorption config |
| 4.4 | ⚪ | Inject absorption line, validate tau recovery | tau recovered within 30% |

---

### Phase 5 - Flexible Tying (V2 only)

| Step | Status | Task | Test |
|------|--------|------|------|
| 5.1 | ⚪ | Add `TyingGroup` dataclass to parameters.py | Unit test dataclass |
| 5.2 | ⚪ | Implement `parse_tying_config()` | Parse sample config correctly |
| 5.3 | ⚪ | Update V2 to use flexible tying matrices | Model runs with new tying |
| 5.4 | ⚪ | Backward compat: legacy config produces same matrices | Matrix diff = 0 |
| 5.5 | ⚪ | Test: tie Ha/Hb z, leave OIII independent | Correct parameter sharing |

---

## Additional TODO Items

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
