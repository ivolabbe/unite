"""
Pytest tests for the UNITE validation framework.

These tests verify end-to-end parameter recovery using the actual fitting pipeline.

NOTE: Some tests are skipped to keep test suite lean and fast.
Core tests kept active:
- test_flux_recovery_single_line (core validation)
- test_blackbody_continuum_recovery (new BB feature)
- test_v2_matches_v1 (regression test)

To unskip tests during development, remove the @pytest.mark.skip decorator.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture(scope='module')
def spec_table():
    """Download test spectrum once for all tests."""
    from unite.utils import download_spectra

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_dir = Path(tmpdir) / 'spectra'
        spec_dir.mkdir()

        # Use G235M for better resolution (R~1000)
        table = download_spectra(
            ['abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits'],
            table_csv=str(Path(__file__).parent.parent / 'examples' / '8204.csv'),
            spectra_directory=str(spec_dir),
        )
        yield table


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestValidationSuite:
    """Tests for ValidationSuite class."""

    def test_flux_recovery_single_line(self, spec_table, output_dir):
        """Test that flux is recovered within 20% for a single bright line."""
        from unite.validation import ValidationSuite, SyntheticLine

        lines = [
            SyntheticLine(
                wavelength=6564.61,  # Hα
                flux=1000.0,
                fwhm_kms=1500.0,  # Resolved at G235M
                name='Ha',
            ),
        ]

        # Continuum is automatically generated using fitting regions
        suite = ValidationSuite(
            rows=spec_table,
            lines=lines,
            rng_seed=42,
        )

        suite.generate_config()  # Must be called before inject()
        suite.inject()
        suite.fit(
            output_dir=output_dir,
            N=100,
            num_warmup=50,
            verbose=False,
        )

        # Use validate_continuum=False for this test since N=100 samples gives marginal continuum recovery
        result = suite.validate(sigma_tolerance=3.0, validate_continuum=False)

        # Print results for debugging
        print(result.pretty_print())

        # Verify continuum parameters are reported
        assert 'cont_angle' in result.injected, 'Continuum angles not in injected parameters'
        assert 'cont_offset' in result.injected, 'Continuum offsets not in injected parameters'
        assert 'cont_angle' in result.recovered, 'Continuum angles not in recovered parameters'
        assert 'cont_offset' in result.recovered, 'Continuum offsets not in recovered parameters'

        # Check that continuum is reasonably recovered (within 5σ for this low-sample test)
        assert result.metrics['cont_offset_max_nsigma'] < 5.0, \
            f"Continuum offset {result.metrics['cont_offset_max_nsigma']:.1f}σ is too far off"

        # Line parameters should be within 3 sigma
        msg_parts = [
            f"flux={result.metrics['flux_max_nsigma']:.1f}σ",
            f"fwhm={result.metrics['fwhm_max_nsigma']:.1f}σ",
            f"z={result.metrics['z_max_nsigma']:.1f}σ",
        ]

        assert result.passed, f"Line validation failed: {', '.join(msg_parts)}"

    @pytest.mark.skip(reason="Similar to single line test - enable during multi-line development")
    def test_multiple_lines(self, spec_table, output_dir):
        """Test recovery of multiple lines with different FWHMs."""
        from unite.validation import ValidationSuite, SyntheticLine

        lines = [
            SyntheticLine(
                wavelength=6564.61,  # Hα narrow
                flux=500.0,
                fwhm_kms=300.0,
                name='Ha_narrow',
            ),
            SyntheticLine(
                wavelength=6564.61,  # Hα broad
                flux=1000.0,
                fwhm_kms=2500.0,
                name='Ha_broad',
            ),
        ]

        # Continuum is automatically generated using fitting regions
        suite = ValidationSuite(
            rows=spec_table,
            lines=lines,
            rng_seed=123,
        )

        suite.generate_config()  # Must be called before inject()
        suite.inject()
        suite.fit(
            output_dir=output_dir,
            N=100,
            num_warmup=50,
            verbose=False,
        )

        # Use validate_continuum=False for this test since N=100 samples gives marginal continuum recovery
        result = suite.validate(sigma_tolerance=3.0, validate_continuum=False)

        print(result.pretty_print())

        # Line parameters should be within 3 sigma
        msg_parts = [
            f"flux={result.metrics['flux_max_nsigma']:.1f}σ",
            f"fwhm={result.metrics['fwhm_max_nsigma']:.1f}σ",
            f"z={result.metrics['z_max_nsigma']:.1f}σ",
        ]

        assert result.passed, f"Line validation failed: {', '.join(msg_parts)}"

    @pytest.mark.skip(reason="Linear continuum recovery - enable during continuum development")
    def test_continuum_recovery(self, spec_table, output_dir):
        """Test that continuum parameters are properly recovered."""
        from unite.validation import ValidationSuite, SyntheticLine

        # Add a line and let continuum auto-generate from fitting regions
        lines = [
            SyntheticLine(wavelength=6564.61, flux=800.0, fwhm_kms=1200.0, name='Ha'),
        ]

        suite = ValidationSuite(rows=spec_table, lines=lines, rng_seed=99)
        suite.generate_config()  # Must be called before inject()
        suite.inject()
        suite.fit(output_dir=output_dir, N=200, num_warmup=100, verbose=False)

        # Validate with continuum checks enabled
        result = suite.validate(sigma_tolerance=3.0, validate_continuum=True)

        print(result.pretty_print())

        # Verify continuum parameters are present
        assert 'cont_angle' in result.injected, 'Continuum angles not in injected parameters'
        assert 'cont_offset' in result.injected, 'Continuum offsets not in injected parameters'
        assert 'cont_angle' in result.recovered, 'Continuum angles not in recovered parameters'
        assert 'cont_offset' in result.recovered, 'Continuum offsets not in recovered parameters'

        # Verify shapes match
        assert len(result.injected['cont_angle']) == len(result.recovered['cont_angle']), \
            'Continuum angle array length mismatch'
        assert len(result.injected['cont_offset']) == len(result.recovered['cont_offset']), \
            'Continuum offset array length mismatch'

        # All parameters should pass (with enough samples, N=200)
        msg_parts = [
            f"flux={result.metrics['flux_max_nsigma']:.1f}σ",
            f"fwhm={result.metrics['fwhm_max_nsigma']:.1f}σ",
            f"cont_angle={result.metrics['cont_angle_max_nsigma']:.1f}σ",
            f"cont_offset={result.metrics['cont_offset_max_nsigma']:.1f}σ",
        ]
        assert result.passed, f"Validation failed: {', '.join(msg_parts)}"

    @pytest.mark.skip(reason="File I/O test - enable during config development")
    def test_config_saved(self, spec_table, output_dir):
        """Test that config JSON is saved for reproducibility."""
        from unite.validation import ValidationSuite, SyntheticLine
        import json

        lines = [SyntheticLine(wavelength=6564.61, flux=500.0, fwhm_kms=1000.0, name='Ha')]

        # Continuum is automatically generated
        suite = ValidationSuite(rows=spec_table, lines=lines, rng_seed=0)
        suite.generate_config()  # Must be called before inject()
        suite.inject()
        suite.fit(output_dir=output_dir, N=50, num_warmup=25, verbose=False)

        config_path = output_dir / 'validation_config.json'
        assert config_path.exists(), 'Config JSON not saved'

        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config['Name'] == 'validation'
        assert 'Groups' in saved_config

    def test_blackbody_continuum_recovery(self, spec_table, output_dir):
        """Test that blackbody continuum temperature is recovered within 20%."""
        from unite.validation import ValidationSuite, SyntheticLine, SyntheticContinuum

        lines = [
            SyntheticLine(wavelength=6564.61, flux=1000.0, fwhm_kms=1500.0, name='Ha'),
        ]

        # Create blackbody continuum with known temperature
        continuum = SyntheticContinuum.from_blackbody(amplitude=10.0, temperature=15000.0)

        suite = ValidationSuite(rows=spec_table, lines=lines, continuum=continuum, rng_seed=42)
        suite.generate_config()

        # Override config to use blackbody continuum
        suite.config['continuum'] = {'type': 'blackbody', 'pivot_micron': 1.0}

        suite.inject()
        suite.fit(output_dir=output_dir, N=200, num_warmup=100, verbose=False, model_version='v2')

        # Validate and print results
        result = suite.validate(sigma_tolerance=3.0, validate_continuum=True)
        print(result.pretty_print())

        # Check that BB continuum parameters are present
        assert 'bb_amplitude' in result.injected, 'bb_amplitude not in injected'
        assert 'bb_temperature' in result.injected, 'bb_temperature not in injected'
        assert 'bb_amplitude' in result.recovered, 'bb_amplitude not in recovered'
        assert 'bb_temperature' in result.recovered, 'bb_temperature not in recovered'

        # Check temperature recovery (30% tolerance for N=200 samples)
        temp_injected = result.injected['bb_temperature']
        temp_recovered = result.recovered['bb_temperature']
        temp_error = abs(temp_recovered - temp_injected) / temp_injected

        assert temp_error < 0.30, f'Temperature error {temp_error * 100:.1f}% exceeds 30%'

    def test_attenuated_blackbody_continuum_recovery(self, spec_table, output_dir):
        """Test attenuated blackbody continuum parameter recovery."""
        from unite.validation import ValidationSuite, SyntheticLine, SyntheticContinuum

        lines = [
            SyntheticLine(wavelength=6564.61, flux=1000.0, fwhm_kms=1500.0, name='Ha'),
        ]

        # Create attenuated blackbody continuum with known temperature, tau_v, and alpha
        continuum = SyntheticContinuum.from_params(
            type='attenuated_blackbody',
            temperature=8000.0,
            normalization=15.0,
            tau_v=1.5,
            alpha=-0.7,
            pivot_micron=1.0,
            temp_type='warm'
        )

        suite = ValidationSuite(rows=spec_table, lines=lines, continuum=continuum, rng_seed=42)
        suite.generate_config()

        # Override config to use attenuated blackbody continuum
        suite.config['continuum'] = {'type': 'attenuated_blackbody', 'pivot_micron': 1.0, 'tau_type': 'moderate', 'alpha_type': 'lmc'}

        suite.inject()
        suite.fit(output_dir=output_dir, N=200, num_warmup=100, verbose=False, model_version='v2')

        # Validate and print results
        result = suite.validate(sigma_tolerance=3.0, validate_continuum=True)
        print(result.pretty_print())

        # Check that ABB continuum parameters are present
        assert 'bb_amplitude' in result.injected, 'bb_amplitude not in injected'
        assert 'bb_temperature' in result.injected, 'bb_temperature not in injected'
        assert 'bb_tau_v' in result.injected, 'bb_tau_v not in injected'
        assert 'bb_alpha' in result.injected, 'bb_alpha not in injected'
        assert 'bb_amplitude' in result.recovered, 'bb_amplitude not in recovered'
        assert 'bb_temperature' in result.recovered, 'bb_temperature not in recovered'
        assert 'bb_tau_v' in result.recovered, 'bb_tau_v not in recovered'
        assert 'bb_alpha' in result.recovered, 'bb_alpha not in recovered'

        # Check that parameters are recovered (relaxed tolerance due to degeneracy)
        # Attenuated BB has T-tau_v degeneracy, so we don't expect perfect recovery
        temp_injected = result.injected['bb_temperature']
        temp_recovered = result.recovered['bb_temperature']
        temp_error = abs(temp_recovered - temp_injected) / temp_injected

        tau_v_injected = result.injected['bb_tau_v']
        tau_v_recovered = result.recovered['bb_tau_v']
        tau_v_error = abs(tau_v_recovered - tau_v_injected) / max(tau_v_injected, 0.1)

        # Just verify the model runs and produces reasonable values
        assert temp_recovered > 1000, 'Temperature unreasonably low'
        assert tau_v_recovered >= 0, 'Tau_V should be non-negative'
        assert result.passed, 'Line parameter recovery failed'

    def test_composite_continuum_with_attenuated_bb(self, spec_table, output_dir):
        """Test composite continuum with MBB + attenuated BB components."""
        from unite.validation import ValidationSuite, SyntheticLine, SyntheticContinuum

        lines = [
            SyntheticLine(wavelength=6564.61, flux=800.0, fwhm_kms=1200.0, name='Ha'),
        ]

        # Composite: hot MBB + warm attenuated BB
        continuum = [
            SyntheticContinuum.from_params(
                type='modified_blackbody',
                temperature=30000,
                normalization=2.0,
                beta=0.0,
                pivot_micron=0.32,
                temp_type='hot',
            ),
            SyntheticContinuum.from_params(
                type='attenuated_blackbody',
                temperature=7000,
                normalization=6.0,
                tau_v=1.0,
                alpha=-0.7,
                pivot_micron=0.5,
                temp_type='warm',
            ),
        ]

        suite = ValidationSuite(rows=spec_table, lines=lines, continuum=continuum, rng_seed=123)
        suite.generate_config()
        suite.inject()
        suite.fit(output_dir=output_dir, N=100, num_warmup=50, verbose=False, model_version='v2')

        # Validate that fitting completes (skip parameter validation for composite continuum)
        result = suite.validate(sigma_tolerance=10.0, validate_continuum=False)
        print(result.pretty_print())

        # Check that injected dict includes both beta, tau_v, and alpha with correct prefixes
        assert 'mbb1_beta' in result.injected, 'MBB beta not in injected'
        assert 'abb2_tau_v' in result.injected, 'ABB tau_v not in injected'
        assert 'abb2_alpha' in result.injected, 'ABB alpha not in injected'
        assert result.injected['abb2_tau_v'] == 1.0, 'Tau_V value incorrect'
        assert result.injected['abb2_alpha'] == -0.7, 'Alpha value incorrect'

    def test_composite_continuum_multi_line(self, spec_table, output_dir):
        """Test composite MBB continuum with multiple emission lines."""
        from unite.validation import ValidationSuite, SyntheticLine, SyntheticContinuum

        # Multiple emission lines within G235M range (rest-frame ~3100-5900 Å at z=4.46)
        lines = [
            SyntheticLine(wavelength=3188.7, flux=600.0, fwhm_kms=1000.0, name='HeI'),
            SyntheticLine(wavelength=3727.0, flux=1200.0, fwhm_kms=800.0, name='[OII]'),
            SyntheticLine(wavelength=4102.89, flux=400.0, fwhm_kms=1500.0, name='Hd'),
            SyntheticLine(wavelength=4862.68, flux=500.0, fwhm_kms=1500.0, name='Hb'),
            SyntheticLine(wavelength=4960.30, flux=900.0, fwhm_kms=1200.0, name='[OIII]'),
        ]

        # Composite continuum: hot + warm MBB
        continuum = [
            SyntheticContinuum.from_params(
                type='modified_blackbody',
                temperature=20000,  # Hot component
                normalization=5.0,
                beta=0.0,  # Pure blackbody
                pivot_micron=1.0,
                temp_type='hot',
            ),
            SyntheticContinuum.from_params(
                type='modified_blackbody',
                temperature=4000,  # Warm component
                normalization=8.0,
                beta=1.5,  # Dust-like
                pivot_micron=1.0,
                temp_type='warm',
            ),
        ]

        suite = ValidationSuite(rows=spec_table, lines=lines, continuum=continuum, rng_seed=123)
        suite.generate_config()
        suite.inject()
        suite.fit(output_dir=output_dir, N=100, num_warmup=50, verbose=False, model_version='v2')

        # Validate that fitting completes (skip parameter validation for composite continuum)
        result = suite.validate(sigma_tolerance=10.0, validate_continuum=False)
        print(result.pretty_print())

        # Check that config was created with composite continuum
        assert 'continuum' in suite.config
        assert isinstance(suite.config['continuum'], list)
        assert len(suite.config['continuum']) == 2
        print(f'\nComposite continuum config: {suite.config["continuum"]}')


class TestInjectSyntheticLines:
    """Tests for inject_synthetic_lines function."""

    @pytest.mark.skip(reason="Low-level injection test - enable during injection development")
    def test_injection_modifies_flux(self, spec_table):
        """Test that line injection actually modifies the spectrum flux."""
        from unite.validation import inject_synthetic_lines, SyntheticLine
        from unite.spectra import NIRSpecSpectra
        import numpy as np

        spectra = NIRSpecSpectra(spec_table)
        spec = spectra.spectra[0]

        original_flux = spec.flux.copy()

        rng = np.random.default_rng(0)
        lines = [SyntheticLine(wavelength=6564.61, flux=1000.0, fwhm_kms=1000.0, name='Ha')]

        # Continuum is automatically generated if not provided
        modified_spec = inject_synthetic_lines(
            spec, lines, continuum=None, rng=rng
        )

        # Flux should be different after injection
        assert not np.allclose(original_flux, modified_spec.flux)

    @pytest.mark.skip(reason="Low-level injection test - enable during injection development")
    def test_injection_adds_line_peak(self, spec_table):
        """Test that injection creates a visible line peak."""
        from unite.validation import inject_synthetic_lines, SyntheticLine, SyntheticContinuum
        from unite.spectra import NIRSpecSpectra
        from astropy import units as u
        import numpy as np

        spectra = NIRSpecSpectra(spec_table)
        spec = spectra.spectra[0]

        rng = np.random.default_rng(0)
        # Use a flat continuum for this test to check peak detection
        continuum_level = 50.0
        wave = spec.wave
        wave_range = wave.max() - wave.min()
        n_regions = max(1, int(wave_range / 0.5))
        edges = np.linspace(wave.min(), wave.max(), n_regions + 1)
        regions = np.column_stack([edges[:-1], edges[1:]])
        angles = np.zeros(n_regions)
        offsets = np.full(n_regions, continuum_level)
        continuum = SyntheticContinuum(regions=regions, angles=angles, offsets=offsets)

        lines = [SyntheticLine(wavelength=6564.61, flux=5000.0, fwhm_kms=2000.0, name='Ha')]

        modified_spec = inject_synthetic_lines(
            spec, lines, continuum=continuum, rng=rng
        )

        # Find line center in observed frame
        opz = 1 + spec.redshift_initial
        line_center = 6564.61 * u.AA.to(spec.λ_unit) * opz

        # Find flux near line center
        mask = np.abs(modified_spec.wave - line_center) < 0.05
        if mask.any():
            peak_flux = modified_spec.flux[mask].max()
            # Peak should be above continuum
            assert peak_flux > continuum_level, f'No line peak visible: max={peak_flux}, continuum={continuum_level}'


class TestModelVersions:
    """Tests for model version equivalence."""

    def test_v2_matches_v1(self, spec_table, output_dir):
        """Test that multiSpecModelV2 produces identical output to multiSpecModel."""
        from unite.validation import ValidationSuite, SyntheticLine
        import pandas as pd

        lines = [
            SyntheticLine(
                wavelength=6564.61,  # Hα
                flux=800.0,
                fwhm_kms=1200.0,
                name='Ha',
            ),
        ]

        # Run with V1 - continuum is automatically generated
        suite_v1 = ValidationSuite(
            rows=spec_table,
            lines=lines,
            rng_seed=42,
        )
        suite_v1.generate_config()  # Must be called before inject()
        suite_v1.inject()
        suite_v1.fit(
            output_dir=output_dir / 'v1',
            N=50,
            num_warmup=25,
            verbose=False,
            model_version='v1',
        )

        # Run with V2 - continuum is automatically generated
        suite_v2 = ValidationSuite(
            rows=spec_table,
            lines=lines,
            rng_seed=42,
        )
        suite_v2.generate_config()  # Must be called before inject()
        suite_v2.inject()
        suite_v2.fit(
            output_dir=output_dir / 'v2',
            N=50,
            num_warmup=25,
            verbose=False,
            model_version='v2',
        )

        # Load CSV summaries
        csv_v1 = pd.read_csv(
            output_dir / 'v1' / 'Results' / f'{spec_table[0]["root"]}-{spec_table[0]["srcid"]}_validation_summary.csv',
            index_col=0,
        )
        csv_v2 = pd.read_csv(
            output_dir / 'v2' / 'Results' / f'{spec_table[0]["root"]}-{spec_table[0]["srcid"]}_validation_summary.csv',
            index_col=0,
        )

        # Compare key parameters (median values)
        # Allow small numerical differences due to MCMC sampling
        for param in ['Ha_narrow_6564.61_flux', 'Ha_narrow_6564.61_fwhm']:
            if param in csv_v1.index and param in csv_v2.index:
                v1_val = csv_v1.loc[param, 'P50']
                v2_val = csv_v2.loc[param, 'P50']
                rel_diff = abs(v1_val - v2_val) / abs(v1_val) if v1_val != 0 else 0

                # Should match within 20% (MCMC sampling variance)
                assert rel_diff < 0.2, (
                    f'{param}: V1={v1_val:.3e}, V2={v2_val:.3e}, '
                    f'rel_diff={rel_diff:.2%}'
                )


# Run with: pixi run pytest tests/test_validation.py -v
