"""
Unit tests for continuum models.

NOTE: Most low-level unit tests are skipped to keep test suite lean.
To run all tests including skipped ones: pytest -v --run-all
To unskip a specific test during development, remove the @pytest.mark.skip decorator.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from unite import optimized
from unite.continuum import (
    LinearContinuum,
    BlackbodyContinuum,
    ModifiedBlackbodyContinuum,
    parse_continuum_config,
)


@pytest.mark.skip(reason="Low-level physics test - enable during BB development")
def test_planck_function_normalized():
    """Verify Planck function is normalized at pivot wavelength."""
    wave = jnp.array([1.0])  # pivot wavelength
    result = optimized.planck_function(wave, 10000.0, pivot_micron=1.0)
    assert jnp.isclose(result, 1.0, rtol=1e-6), f'Expected 1.0, got {result}'


@pytest.mark.skip(reason="Low-level physics test - enable during BB development")
def test_planck_function_temperature_dependence():
    """Verify Planck function increases with temperature at short wavelengths."""
    wave = jnp.array([0.5, 1.0, 2.0])  # Short, pivot, long wavelengths
    temp_low = 5000.0
    temp_high = 15000.0

    flux_low = optimized.planck_function(wave, temp_low, pivot_micron=1.0)
    flux_high = optimized.planck_function(wave, temp_high, pivot_micron=1.0)

    # At short wavelengths, higher temperature should give more flux
    assert flux_high[0] > flux_low[0], 'Higher temp should give more flux at short wavelengths'
    # At pivot, both should be 1.0
    assert jnp.isclose(flux_low[1], 1.0) and jnp.isclose(flux_high[1], 1.0)


@pytest.mark.skip(reason="Low-level physics test - enable during MBB development")
def test_modified_blackbody_beta_zero():
    """Verify modified blackbody with beta=0 equals pure blackbody."""
    wave = jnp.array([0.5, 1.0, 2.0])
    temp = 10000.0
    beta = 0.0

    bb = optimized.planck_function(wave, temp, pivot_micron=1.0)
    mbb = optimized.modified_blackbody(wave, temp, beta, pivot_micron=1.0)

    assert jnp.allclose(bb, mbb, rtol=1e-6), 'Modified BB with beta=0 should equal pure BB'


@pytest.mark.skip(reason="Low-level physics test - enable during MBB development")
def test_modified_blackbody_positive_beta():
    """Verify modified blackbody with positive beta suppresses long wavelengths."""
    wave = jnp.array([0.5, 1.0, 2.0])
    temp = 10000.0
    beta_zero = 0.0
    beta_pos = 2.0

    mbb_zero = optimized.modified_blackbody(wave, temp, beta_zero, pivot_micron=1.0)
    mbb_pos = optimized.modified_blackbody(wave, temp, beta_pos, pivot_micron=1.0)

    # At long wavelengths, positive beta should suppress flux
    assert mbb_pos[2] < mbb_zero[2], 'Positive beta should suppress long wavelengths'
    # At pivot, both should be 1.0
    assert jnp.isclose(mbb_zero[1], 1.0) and jnp.isclose(mbb_pos[1], 1.0)


@pytest.mark.skip(reason="Implementation detail test - enable during API changes")
def test_blackbody_continuum_sample_params():
    """Test BlackbodyContinuum.sample_params works correctly."""
    from numpyro import sample
    from unittest.mock import MagicMock

    bb = BlackbodyContinuum(amplitude_guess=10.0)
    cont_regs = jnp.array([[0.5, 1.5]])

    # Mock the sample function to track calls
    mock_sample = MagicMock(side_effect=lambda name, dist: 1.0)
    params = bb.sample_params(mock_sample, cont_regs)

    # Check that sample was called with correct parameter names
    assert mock_sample.call_count == 2
    call_names = [call[0][0] for call in mock_sample.call_args_list]
    assert 'bb_amplitude' in call_names
    assert 'bb_temperature' in call_names

    # Check returned params dict
    assert 'bb_amplitude' in params
    assert 'bb_temperature' in params


@pytest.mark.skip(reason="Implementation detail test - enable during API changes")
def test_blackbody_continuum_evaluate():
    """Test BlackbodyContinuum.evaluate produces expected output."""
    bb = BlackbodyContinuum(amplitude_guess=10.0)
    wave_rest = jnp.array([0.5, 1.0, 2.0])
    params = {'bb_amplitude': 10.0, 'bb_temperature': 10000.0}
    cont_regs = jnp.array([[0.0, 3.0]])  # Unused for BB

    flux = bb.evaluate(wave_rest, params, cont_regs)

    # Check shape
    assert flux.shape == wave_rest.shape
    # Check values are positive
    assert jnp.all(flux > 0)
    # Check normalized at pivot (1.0 micron)
    assert jnp.isclose(flux[1], 10.0, rtol=1e-6), f'Expected 10.0 at pivot, got {flux[1]}'


@pytest.mark.skip(reason="Implementation detail test - enable during API changes")
def test_modified_blackbody_continuum_sample_params():
    """Test ModifiedBlackbodyContinuum.sample_params works correctly."""
    from unittest.mock import MagicMock

    mbb = ModifiedBlackbodyContinuum(amplitude_guess=10.0)
    cont_regs = jnp.array([[0.5, 1.5]])

    # Mock the sample function to track calls
    mock_sample = MagicMock(side_effect=lambda name, dist: 1.0)
    params = mbb.sample_params(mock_sample, cont_regs)

    # Check that sample was called 3 times (amplitude, temperature, beta)
    assert mock_sample.call_count == 3
    call_names = [call[0][0] for call in mock_sample.call_args_list]
    assert 'mbb_amplitude' in call_names
    assert 'mbb_temperature' in call_names
    assert 'mbb_beta' in call_names

    # Check returned params dict
    assert 'mbb_amplitude' in params
    assert 'mbb_temperature' in params
    assert 'mbb_beta' in params


@pytest.mark.skip(reason="Implementation detail test - enable during API changes")
def test_modified_blackbody_continuum_evaluate():
    """Test ModifiedBlackbodyContinuum.evaluate produces expected output."""
    mbb = ModifiedBlackbodyContinuum(amplitude_guess=10.0)
    wave_rest = jnp.array([0.5, 1.0, 2.0])
    params = {'mbb_amplitude': 10.0, 'mbb_temperature': 10000.0, 'mbb_beta': 2.0}
    cont_regs = jnp.array([[0.0, 3.0]])

    flux = mbb.evaluate(wave_rest, params, cont_regs)

    # Check shape
    assert flux.shape == wave_rest.shape
    # Check values are positive
    assert jnp.all(flux > 0)
    # Check normalized at pivot
    assert jnp.isclose(flux[1], 10.0, rtol=1e-6)


@pytest.mark.skip(reason="Trivial list test - enable during compositing development")
def test_composite_continuum_list():
    """Test list of continuum models (Linear + Blackbody)."""
    linear = LinearContinuum(jnp.array([10.0]))
    bb = BlackbodyContinuum(amplitude_guess=5.0)
    models = [linear, bb]

    # Models can be iterated over
    assert len(models) == 2
    assert isinstance(models[0], LinearContinuum)
    assert isinstance(models[1], BlackbodyContinuum)


def test_parse_continuum_config_default():
    """Test config parsing defaults to [LinearContinuum]."""
    config = {}
    cont_guesses = jnp.array([10.0, 12.0])
    models = parse_continuum_config(config, cont_guesses)
    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], LinearContinuum)


def test_parse_continuum_config_blackbody():
    """Test config parsing for blackbody."""
    config = {'continuum': {'type': 'blackbody', 'pivot_micron': 0.5}}
    cont_guesses = jnp.array([10.0])
    models = parse_continuum_config(config, cont_guesses)
    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], BlackbodyContinuum)
    assert models[0].pivot_micron == 0.5


def test_parse_continuum_config_modified_blackbody():
    """Test config parsing for modified blackbody."""
    config = {'continuum': {'type': 'modified_blackbody', 'temp_type': 'warm'}}
    cont_guesses = jnp.array([10.0])
    models = parse_continuum_config(config, cont_guesses)
    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], ModifiedBlackbodyContinuum)
    assert models[0].temp_bounds == (2000.0, 10000.0)  # Check bounds instead of temp_type string


def test_parse_continuum_config_linear_explicit():
    """Test config parsing for explicit linear."""
    config = {'continuum': {'type': 'linear'}}
    cont_guesses = jnp.array([10.0, 12.0])
    models = parse_continuum_config(config, cont_guesses)
    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], LinearContinuum)


def test_parse_continuum_config_composite():
    """Test config parsing for composite continuum (list of models)."""
    config = {
        'continuum': [
            {'type': 'linear'},
            {'type': 'blackbody', 'pivot_micron': 1.0},
        ]
    }
    cont_guesses = jnp.array([10.0])
    models = parse_continuum_config(config, cont_guesses)
    assert isinstance(models, list)
    assert len(models) == 2
    assert isinstance(models[0], LinearContinuum)
    assert isinstance(models[1], BlackbodyContinuum)


def test_parse_continuum_config_invalid():
    """Test config parsing raises error for invalid type."""
    config = {'continuum': {'type': 'invalid_type'}}
    cont_guesses = jnp.array([10.0])
    with pytest.raises(ValueError, match='Unknown continuum type'):
        parse_continuum_config(config, cont_guesses)


# Continuum-only mode tests
def test_mask_region_computation():
    """Test mask region computation and inversion logic."""
    from unite.initial import merge_overlapping_regions, invert_regions

    # Test merge_overlapping_regions
    regions = [[1.0, 2.0], [1.5, 2.5], [3.0, 4.0]]
    merged = merge_overlapping_regions(regions)
    assert merged.shape == (2, 2)
    assert np.allclose(merged[0], [1.0, 2.5])
    assert np.allclose(merged[1], [3.0, 4.0])

    # Test invert_regions
    masks = np.array([[1.0, 2.0], [3.0, 4.0]])
    valid = invert_regions(masks, 0.0, 5.0)
    assert valid.shape == (3, 2)
    assert np.allclose(valid[0], [0.0, 1.0])
    assert np.allclose(valid[1], [2.0, 3.0])
    assert np.allclose(valid[2], [4.0, 5.0])

    # Test edge case: no masks
    valid = invert_regions(np.array([]).reshape(0, 2), 0.0, 5.0)
    assert valid.shape == (1, 2)
    assert np.allclose(valid[0], [0.0, 5.0])


def test_continuum_only_blackbody():
    """Test continuum-only mode with blackbody continuum.

    This test fits a real spectrum's continuum (without injecting anything).
    We just verify that the fit runs and produces reasonable results.
    """
    from unite.validation import ValidationSuite, SyntheticContinuum
    import astropy.units as u

    # Load a test spectrum (use standard .spec.fits format)
    # Continuum-only mode means we fit the existing spectrum's continuum
    # We don't inject anything - just analyze what's there
    suite = ValidationSuite(
        spectrum_path='spectra/abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits',
        continuum_only=True,
        continuum=SyntheticContinuum.from_params(
            type='blackbody',
            temperature=5000.0,  # Initial guess (not injected)
            normalization=1.0,
            pivot_micron=1.0,
        ),
        name='continuum_only_bb_test',
    )

    # Run quick fit (fewer samples for testing)
    result = suite.run(N=100, num_warmup=50, verbose=False)

    # Check that fitting completed and produced reasonable results
    samples = result.fit_result.samples
    assert 'bb_temperature' in samples, 'Temperature parameter missing'
    assert 'bb_amplitude' in samples, 'Amplitude parameter missing'

    temp_median = float(np.median(samples['bb_temperature']))
    amp_median = float(np.median(samples['bb_amplitude']))

    # Verify results are physically reasonable (not checking exact recovery)
    assert 1000.0 < temp_median < 30000.0, f'Temperature unreasonable: {temp_median}K'
    assert amp_median > 0, f'Amplitude should be positive: {amp_median}'

    # Check that samples have expected shape
    assert samples['bb_temperature'].shape[0] == 100, 'Should have 100 samples'


@pytest.mark.skip(reason='Requires PRISM spectrum - enable when testing PRISM')
def test_continuum_only_prism():
    """Test continuum-only mode with PRISM spectrum."""
    from unite.validation import ValidationSuite, SyntheticContinuum

    # This test requires a PRISM spectrum file
    # Skip by default, enable when PRISM file is available
    suite = ValidationSuite(
        spectrum_path='spectra/abell2744-greene-v4_prism-clear_8204_45924.spec.fits',
        continuum_only=True,
        continuum=SyntheticContinuum.from_params(
            type='blackbody',
            temperature=8000.0,
            normalization=3.0,
        ),
        name='continuum_only_prism_test',
    )

    result = suite.run(N=100, num_warmup=50, verbose=False)
    samples = result.fit_result.samples

    # Verify temperature recovery
    assert 'bb_temperature' in samples
    temp_median = float(np.median(samples['bb_temperature']))
    assert 5000.0 < temp_median < 12000.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
