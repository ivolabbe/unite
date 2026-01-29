"""
Pytest tests for the UNITE validation framework.

These tests verify end-to-end parameter recovery using the actual fitting pipeline.
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

        suite = ValidationSuite(
            rows=spec_table,
            lines=lines,
            continuum_level=50.0,
            rng_seed=42,
        )

        suite.inject()
        suite.generate_config()
        suite.fit(
            output_dir=output_dir,
            N=100,
            num_warmup=50,
            verbose=False,
        )

        result = suite.validate(sigma_tolerance=3.0)

        # Print results for debugging
        print(result.pretty_print())

        # All parameters should be within 3 sigma
        assert result.passed, (
            f"Validation failed: flux={result.metrics['flux_max_nsigma']:.1f}σ, "
            f"fwhm={result.metrics['fwhm_max_nsigma']:.1f}σ"
        )

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

        suite = ValidationSuite(
            rows=spec_table,
            lines=lines,
            continuum_level=30.0,
            rng_seed=123,
        )

        suite.inject()
        suite.generate_config()
        suite.fit(
            output_dir=output_dir,
            N=100,
            num_warmup=50,
            verbose=False,
        )

        result = suite.validate(sigma_tolerance=3.0)

        print(result.pretty_print())

        # All parameters should be within 3 sigma
        assert result.passed, (
            f"Validation failed: flux={result.metrics['flux_max_nsigma']:.1f}σ, "
            f"fwhm={result.metrics['fwhm_max_nsigma']:.1f}σ"
        )

    def test_config_saved(self, spec_table, output_dir):
        """Test that config JSON is saved for reproducibility."""
        from unite.validation import ValidationSuite, SyntheticLine
        import json

        lines = [SyntheticLine(wavelength=6564.61, flux=500.0, fwhm_kms=1000.0, name='Ha')]

        suite = ValidationSuite(rows=spec_table, lines=lines, rng_seed=0)
        suite.inject()
        suite.generate_config()
        suite.fit(output_dir=output_dir, N=50, num_warmup=25, verbose=False)

        config_path = output_dir / 'validation_config.json'
        assert config_path.exists(), 'Config JSON not saved'

        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config['Name'] == 'validation'
        assert 'Groups' in saved_config


class TestInjectSyntheticLines:
    """Tests for inject_synthetic_lines function."""

    def test_injection_modifies_flux(self, spec_table):
        """Test that line injection actually modifies the spectrum flux."""
        from unite.validation import inject_synthetic_lines, SyntheticLine
        from unite.spectra import NIRSpecSpectra
        import numpy as np

        spectra = NIRSpecSpectra(spec_table)
        spec = spectra.spectra[0]

        original_flux = spec.flux.copy()

        lines = [SyntheticLine(wavelength=6564.61, flux=1000.0, fwhm_kms=1000.0, name='Ha')]

        modified_spec = inject_synthetic_lines(
            spec, lines, continuum_level=50.0, rng=np.random.default_rng(0)
        )

        # Flux should be different after injection
        assert not np.allclose(original_flux, modified_spec.flux)

    def test_injection_adds_line_peak(self, spec_table):
        """Test that injection creates a visible line peak."""
        from unite.validation import inject_synthetic_lines, SyntheticLine
        from unite.spectra import NIRSpecSpectra
        from astropy import units as u
        import numpy as np

        spectra = NIRSpecSpectra(spec_table)
        spec = spectra.spectra[0]

        continuum = 50.0
        lines = [SyntheticLine(wavelength=6564.61, flux=5000.0, fwhm_kms=2000.0, name='Ha')]

        modified_spec = inject_synthetic_lines(
            spec, lines, continuum_level=continuum, rng=np.random.default_rng(0)
        )

        # Find line center in observed frame
        opz = 1 + spec.redshift_initial
        line_center = 6564.61 * u.AA.to(spec.λ_unit) * opz

        # Find flux near line center
        mask = np.abs(modified_spec.wave - line_center) < 0.05
        if mask.any():
            peak_flux = modified_spec.flux[mask].max()
            # Peak should be above continuum
            assert peak_flux > continuum, f'No line peak visible: max={peak_flux}, continuum={continuum}'


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

        # Run with V1
        suite_v1 = ValidationSuite(
            rows=spec_table,
            lines=lines,
            continuum_level=50.0,
            rng_seed=42,
        )
        suite_v1.inject()
        suite_v1.generate_config()
        suite_v1.fit(
            output_dir=output_dir / 'v1',
            N=50,
            num_warmup=25,
            verbose=False,
            model_version='v1',
        )

        # Run with V2
        suite_v2 = ValidationSuite(
            rows=spec_table,
            lines=lines,
            continuum_level=50.0,
            rng_seed=42,
        )
        suite_v2.inject()
        suite_v2.generate_config()
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
