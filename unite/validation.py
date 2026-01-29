"""
Validation framework for end-to-end testing of UNITE.

This module provides utilities to:
- Inject synthetic emission/absorption lines into real spectra
- Run the actual UNITE fitting pipeline and validate parameter recovery
- Compare recovered parameters against injected values
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from astropy import units as u
from astropy.table import Table
from copy import deepcopy
from jax import numpy as jnp

from unite.spectra import NIRSpecSpectra, NIRSpecSpectrum
from unite import optimized
from unite.model import C  # Speed of light in km/s

logger = logging.getLogger(__name__)


@dataclass
class SyntheticLine:
    """Configuration for a synthetic emission/absorption line.

    Attributes
    ----------
    wavelength : float
        Rest wavelength in Angstrom.
    flux : float
        Integrated line flux in units of 1e-20 erg/s/cm^2.
    fwhm_kms : float
        Full width at half maximum in km/s.
    profile : str
        Line profile type: 'gaussian', 'lorentzian', or 'exponential'.
    is_absorption : bool
        If True, line is absorption (subtracts from continuum).
    name : str
        Line identifier for config generation.
    """

    wavelength: float
    flux: float
    fwhm_kms: float
    profile: str = 'gaussian'
    is_absorption: bool = False
    name: str = ''


@dataclass
class SyntheticContinuum:
    """Configuration for synthetic piecewise linear continuum.

    Attributes
    ----------
    regions : np.ndarray
        Continuum regions (Nc, 2) array of [low, high] wavelength bounds in microns.
    angles : np.ndarray
        Angles for each continuum segment (Nc,) in radians.
    offsets : np.ndarray
        Offset heights for each continuum segment (Nc,) in flux units.
    """

    regions: np.ndarray
    angles: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_spectrum_median(
        cls,
        spec: NIRSpecSpectrum,
        angle_range: Tuple[float, float] = (-0.2, 0.2),
        rng: np.random.Generator | None = None,
    ) -> 'SyntheticContinuum':
        """Generate realistic continuum from spectrum median flux.

        WARNING: This creates regions based on full wavelength range, which may not
        match the continuum regions used during fitting (which are based on line coverage).
        For validation, use from_lines() instead to ensure regions match.

        Parameters
        ----------
        spec : NIRSpecSpectrum
            Spectrum to base continuum on.
        angle_range : Tuple[float, float]
            Range of angles in radians for continuum slope.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        SyntheticContinuum
            Continuum model based on spectrum's flux distribution.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        # Use spectrum's wavelength range to define continuum regions
        wave = spec.wave
        wave_range = wave.max() - wave.min()
        n_regions = max(1, int(wave_range / 0.5))  # ~0.5 micron per region

        # Create evenly spaced regions
        edges = np.linspace(wave.min(), wave.max(), n_regions + 1)
        regions = np.column_stack([edges[:-1], edges[1:]])

        # Generate angles and offsets
        angles = rng.uniform(angle_range[0], angle_range[1], n_regions)

        # Base offsets on median flux level with some variation
        median_flux = float(np.nanmedian(spec.flux))
        offsets = rng.uniform(0.7 * median_flux, 1.3 * median_flux, n_regions)

        return cls(regions=regions, angles=angles, offsets=offsets)

    @classmethod
    def from_config(
        cls,
        config: dict,
        spectra,
        angle_range: Tuple[float, float] = (-0.1, 0.1),
        rng: np.random.Generator | None = None,
    ) -> 'SyntheticContinuum':
        """Generate continuum using the same regions that fitting will use.

        This ensures injected and fitted continuum regions match exactly,
        enabling proper validation of continuum parameters.

        Parameters
        ----------
        config : dict
            UNITE config with line definitions.
        spectra : NIRSpecSpectra
            Spectra object.
        angle_range : Tuple[float, float]
            Range of angles in radians for continuum slope.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        SyntheticContinuum
            Continuum model with regions matching what fitting will use.
        """
        from unite import initial

        if rng is None:
            rng = np.random.default_rng(0)

        # Use the SAME regions that fitting will compute
        cont_regs, cont_guesses = initial.computeContinuumRegions(config, spectra)

        n_regions = len(cont_regs)
        angles = rng.uniform(angle_range[0], angle_range[1], n_regions)

        # Use guesses with some variation
        offsets = cont_guesses * rng.uniform(0.8, 1.2, n_regions)

        return cls(regions=np.array(cont_regs), angles=angles, offsets=offsets)


@dataclass
class ValidationResult:
    """Result of a validation test.

    Attributes
    ----------
    passed : bool
        Overall pass/fail status.
    metrics : dict
        Recovery metrics including mean errors and max nsigma.
    injected : dict
        Injected parameter values.
    recovered : dict
        Recovered parameter values (median of posterior).
    uncertainties : dict
        Uncertainty intervals (16th, 84th percentiles) and 1-sigma values.
    nsigma : dict
        Error in units of sigma for each parameter.
    snr : np.ndarray
        Signal-to-noise ratio of each line.
    line_names : List[str]
        Names of the lines.
    sigma_tolerance : float
        Sigma tolerance used for pass/fail.
    details : str
        Human-readable summary of results.
    """

    passed: bool
    metrics: Dict[str, float]
    injected: Dict[str, np.ndarray]
    recovered: Dict[str, np.ndarray]
    uncertainties: Dict[str, np.ndarray]
    nsigma: Dict[str, np.ndarray]
    snr: np.ndarray
    line_names: List[str]
    sigma_tolerance: float
    details: str

    def pretty_print(self) -> str:
        """Generate a pretty ASCII table of validation results.

        Table shows injected values, recovered values, and uncertainty intervals.
        Summary shows percentage error and sigma offset.

        Returns
        -------
        str
            Formatted ASCII table.
        """
        lines = []
        status = '✓ PASSED' if self.passed else '✗ FAILED'
        lines.append(f'\n{"=" * 80}')
        lines.append(f'  VALIDATION RESULTS: {status}  (tolerance: {self.sigma_tolerance:.0f}σ)')
        lines.append(f'{"=" * 80}')

        # Header
        lines.append(
            f'{"Line":<12} {"Param":<8} {"Injected":>12} {"Recovered":>12} '
            f'{"[16%, 84%]":>22} {"SNR":>6}'
        )
        lines.append('-' * 80)

        n_lines = len(self.line_names)
        for i in range(n_lines):
            name = self.line_names[i]
            snr = self.snr[i]

            # Flux row
            f_inj = self.injected['flux'][i]
            f_rec = self.recovered['flux'][i]
            f_lo = self.uncertainties['flux_lo'][i]
            f_hi = self.uncertainties['flux_hi'][i]
            lines.append(
                f'{name:<12} {"flux":<8} {f_inj:>12.1f} {f_rec:>12.1f} '
                f'[{f_lo:>9.1f}, {f_hi:>9.1f}] {snr:>5.1f}'
            )

            # FWHM row
            w_inj = self.injected['fwhm'][i]
            w_rec = self.recovered['fwhm'][i]
            w_lo = self.uncertainties['fwhm_lo'][i]
            w_hi = self.uncertainties['fwhm_hi'][i]
            lines.append(
                f'{"":<12} {"fwhm":<8} {w_inj:>12.0f} {w_rec:>12.0f} ' f'[{w_lo:>9.0f}, {w_hi:>9.0f}]'
            )

            # Redshift row
            z_inj = self.injected['redshift'][i]
            z_rec = self.recovered['redshift'][i]
            z_lo = self.uncertainties['z_lo'][i]
            z_hi = self.uncertainties['z_hi'][i]
            lines.append(
                f'{"":<12} {"z":<8} {z_inj:>12.5f} {z_rec:>12.5f} ' f'[{z_lo:>9.5f}, {z_hi:>9.5f}]'
            )
            lines.append('-' * 80)

        # Add continuum table if available
        if 'cont_angle' in self.injected:
            # Ensure arrays are numpy arrays with consistent shapes
            cont_angle_inj = np.atleast_1d(self.injected['cont_angle'])
            cont_angle_rec = np.atleast_1d(self.recovered['cont_angle'])
            cont_angle_sigma = np.atleast_1d(self.uncertainties['cont_angle_sigma'])

            cont_offset_inj = np.atleast_1d(self.injected['cont_offset'])
            cont_offset_rec = np.atleast_1d(self.recovered['cont_offset'])
            cont_offset_sigma = np.atleast_1d(self.uncertainties['cont_offset_sigma'])

            n_inj = len(cont_angle_inj)
            n_rec = len(cont_angle_rec)

            # Check for shape mismatch
            if n_inj != n_rec:
                lines.append(
                    f'\n{"Region":<10} {"Param":<8} {"Injected":>12} {"Recovered":>12} {"Sigma":>10}'
                )
                lines.append('-' * 80)
                lines.append(f'⚠️  WARNING: Continuum region mismatch!')
                lines.append(f'    Injected: {n_inj} regions, Recovered: {n_rec} regions')
                lines.append(
                    f'    This happens when fitted continuum regions differ from injected regions.'
                )
                lines.append(f'    Showing first {min(n_inj, n_rec)} region(s) for comparison:')
                lines.append('-' * 80)
                n_regions = min(n_inj, n_rec)
            else:
                lines.append(
                    f'\n{"Region":<10} {"Param":<8} {"Injected":>12} {"Recovered":>12} {"Sigma":>10}'
                )
                lines.append('-' * 80)
                n_regions = n_inj

            for i in range(n_regions):
                # Angle row
                lines.append(
                    f'Region {i:<3} {"angle":<8} {cont_angle_inj[i]:>12.4f} {cont_angle_rec[i]:>12.4f} '
                    f'{cont_angle_sigma[i]:>9.4f}'
                )
                # Offset row
                lines.append(
                    f'{"":<10} {"offset":<8} {cont_offset_inj[i]:>12.2f} {cont_offset_rec[i]:>12.2f} '
                    f'{cont_offset_sigma[i]:>9.2f}'
                )

            lines.append('-' * 80)

        # Summary metrics with percentile error and sigma offset
        lines.append(f'\nSummary:')
        lines.append(
            f'  Flux:     mean error = {self.metrics["flux_mean_error"] * 100:>5.1f}%,  '
            f'max offset = {self.metrics["flux_max_nsigma"]:.1f}σ'
        )
        lines.append(
            f'  FWHM:     mean error = {self.metrics["fwhm_mean_error"] * 100:>5.1f}%,  '
            f'max offset = {self.metrics["fwhm_max_nsigma"]:.1f}σ'
        )
        lines.append(
            f'  Redshift: mean error = {self.metrics["z_mean_error"]:.6f},  '
            f'max offset = {self.metrics["z_max_nsigma"]:.1f}σ'
        )

        # Add continuum summary if available
        if 'cont_angle_max_nsigma' in self.metrics:
            lines.append(
                f'  Continuum angle:  ' f'max offset = {self.metrics["cont_angle_max_nsigma"]:.1f}σ'
            )
            lines.append(
                f'  Continuum offset: ' f'max offset = {self.metrics["cont_offset_max_nsigma"]:.1f}σ'
            )

        lines.append(f'{"=" * 80}\n')

        return '\n'.join(lines)


class ValidationSuite:
    """End-to-end validation framework for UNITE.

    This class provides methods to:
    1. Inject synthetic lines into real spectra
    2. Generate UNITE config from synthetic lines
    3. Run the actual NIRSpecFit pipeline
    4. Validate recovered parameters against injected values

    Parameters
    ----------
    rows : Table
        Astropy table with spectrum metadata (from download_spectra or similar).
    lines : List[SyntheticLine]
        Synthetic lines to inject.
    continuum_level : float, optional
        Continuum level. If None, uses median of existing flux.
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        rows: Table,
        lines: List[SyntheticLine],
        continuum: SyntheticContinuum | float | None = None,
        rng_seed: int = 0,
    ) -> None:
        self.rows = rows
        self.lines = lines
        self.rng = np.random.default_rng(rng_seed)

        # Load spectra from rows
        self.base_spectra = NIRSpecSpectra(rows)

        # Handle continuum - convert float to SyntheticContinuum
        if continuum is None:
            # Will generate simple continuum covering lines in inject()
            self.continuum = None
        elif isinstance(continuum, (int, float)):
            # Backward compatibility: flat continuum
            logger.warning(
                'Using flat continuum. Consider using SyntheticContinuum for realistic continuum model.'
            )
            self.continuum = continuum
        else:
            self.continuum = continuum

        # Will be populated after injection/fitting
        self.injected_spectra: NIRSpecSpectra | None = None
        self.injected_continuum: SyntheticContinuum | None = None
        self.config: dict | None = None
        self.samples: dict | None = None
        self.output_dir: Path | None = None
        self._line_order: List[int] | None = None

    def inject(self, spectrum_idx: int = 0, lsf_scale: float = 1.0) -> NIRSpecSpectra:
        """Inject synthetic lines into a spectrum.

        IMPORTANT: Must call generate_config() BEFORE inject() so that continuum
        regions can be computed from the config.

        Parameters
        ----------
        spectrum_idx : int
            Index of the spectrum to inject into.
        lsf_scale : float
            LSF scaling factor (passed to spec.lsf).

        Returns
        -------
        NIRSpecSpectra
            Modified spectra with synthetic lines injected.
        """
        spectra = deepcopy(self.base_spectra)
        spec = spectra.spectra[spectrum_idx]

        # Generate continuum if needed
        continuum = self.continuum
        if continuum is None:
            # Generate continuum using fitting's regions
            if self.config is None:
                raise ValueError('Must call generate_config() before inject() to compute continuum regions')
            logger.info('Generating continuum using fitting regions from config')
            continuum = SyntheticContinuum.from_config(self.config, self.base_spectra, rng=self.rng)
        elif isinstance(continuum, (int, float)):
            # Convert flat continuum to SyntheticContinuum
            wave = spec.wave
            wave_range = wave.max() - wave.min()
            n_regions = max(1, int(wave_range / 0.5))
            edges = np.linspace(wave.min(), wave.max(), n_regions + 1)
            regions = np.column_stack([edges[:-1], edges[1:]])
            angles = np.zeros(n_regions)
            offsets = np.full(n_regions, float(continuum))
            continuum = SyntheticContinuum(regions=regions, angles=angles, offsets=offsets)

        # Store injected continuum
        self.injected_continuum = continuum

        # Inject lines using the actual LSF from the spectrum
        spectra.spectra[spectrum_idx] = inject_synthetic_lines(
            spec, self.lines, continuum=continuum, rng=self.rng, lsf_scale=lsf_scale
        )

        self.injected_spectra = spectra
        return spectra

    def generate_config(self) -> dict:
        """Generate UNITE config from synthetic lines.

        Returns
        -------
        dict
            UNITE configuration dictionary.
        """
        # Group lines by profile type, tracking original indices
        groups: Dict[str, List[Tuple[int, SyntheticLine]]] = {}
        for idx, line in enumerate(self.lines):
            # Map profile to line type
            if line.is_absorption:
                line_type = 'absorption'
            elif line.profile == 'lorentzian':
                line_type = 'lorentzian'
            elif line.profile == 'exponential':
                line_type = 'exponential'
            elif line.fwhm_kms > 750:
                line_type = 'broad'
            else:
                line_type = 'narrow'

            if line_type not in groups:
                groups[line_type] = []
            groups[line_type].append((idx, line))

        # Track line order for validation mapping
        line_order: List[int] = []

        # Build config structure
        config: dict = {'Name': 'validation', 'Unit': 'AA', 'Groups': {}}

        for line_type, type_lines in groups.items():
            group_name = f'val_{line_type}'
            species_list = []

            for idx, line in type_lines:
                line_order.append(idx)
                name = line.name or f'line_{line.wavelength:.1f}'
                species = {
                    'Name': name,
                    'LineType': line_type,
                    'Lines': [{'Wavelength': line.wavelength, 'RelStrength': None}],
                }
                species_list.append(species)

            config['Groups'][group_name] = {
                'TieRedshift': True,
                'TieDispersion': True,
                'Species': species_list,
            }

        self._line_order = line_order
        self.config = config
        return config

    def fit(
        self,
        output_dir: str | Path = 'validation_out',
        N: int = 500,
        num_warmup: int = 250,
        rescale_errors: bool = False,
        verbose: bool = True,
        save_config: bool = True,
        model_version: str = 'v2',
    ) -> dict:
        """Run the actual UNITE fitting pipeline on injected spectra.

        Parameters
        ----------
        output_dir : str or Path
            Directory for output files.
        N : int
            Number of MCMC samples.
        num_warmup : int
            Number of warmup samples.
        rescale_errors : bool
            Whether to rescale errors.
        verbose : bool
            Print progress.
        save_config : bool
            Save config JSON to output directory.
        model_version : str
            Model version to use ('v1' or 'v2'), default 'v2'.

        Returns
        -------
        dict
            MCMC samples dictionary.
        """
        from unite.fitting import NIRSpecFit

        if self.injected_spectra is None:
            self.inject()

        if self.config is None:
            self.generate_config()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility
        if save_config:
            config_path = self.output_dir / 'validation_config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f'Saved config to {config_path}')

        # Run the actual NIRSpecFit pipeline
        NIRSpecFit(
            config=self.config,
            rows=self.rows,
            spectra=self.injected_spectra,
            output_directory=str(self.output_dir),
            N=N,
            num_warmup=num_warmup,
            rescale_errors=rescale_errors,
            verbose=verbose,
            model_version=model_version,
        )

        # Load samples from saved results
        cname = '_' + self.config['Name'] if self.config['Name'] else ''
        results_path = (
            self.output_dir / 'Results' / f'{self.rows[0]["root"]}-{self.rows[0]["srcid"]}{cname}_full.npz'
        )
        self.samples = dict(np.load(results_path))

        return self.samples

    def validate(self, sigma_tolerance: float = 3.0, validate_continuum: bool = True) -> ValidationResult:
        """Validate recovered parameters against injected values.

        Pass/fail is determined by whether the error is within N sigma
        of the recovered uncertainty.

        Parameters
        ----------
        sigma_tolerance : float
            Number of sigma within which recovery is considered passing.
            Default: 3.0 (i.e., error must be < 3σ of the uncertainty).
        validate_continuum : bool
            If True, include continuum parameters in validation checks.
            Default: True. Set to False to only validate line parameters.

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status and metrics.
        """
        if self.samples is None:
            raise ValueError('Must run fit() before validate()')
        if self._line_order is None:
            raise ValueError('Must run generate_config() before validate()')

        # Get recovered parameters (median and percentiles)
        flux_all = self.samples['flux_all']
        fwhm_all = self.samples['fwhm_all']
        z_all = self.samples['redshift_all']

        flux_recovered = np.median(flux_all, axis=0)
        fwhm_recovered = np.median(fwhm_all, axis=0)
        z_recovered = np.median(z_all, axis=0)

        # Uncertainty intervals (16th and 84th percentiles = 1 sigma)
        flux_lo = np.percentile(flux_all, 16, axis=0)
        flux_hi = np.percentile(flux_all, 84, axis=0)
        fwhm_lo = np.percentile(fwhm_all, 16, axis=0)
        fwhm_hi = np.percentile(fwhm_all, 84, axis=0)
        z_lo = np.percentile(z_all, 16, axis=0)
        z_hi = np.percentile(z_all, 84, axis=0)

        # Compute 1-sigma uncertainties (half of 16-84 interval)
        flux_sigma = (flux_hi - flux_lo) / 2
        fwhm_sigma = (fwhm_hi - fwhm_lo) / 2
        z_sigma = (z_hi - z_lo) / 2

        # Get injected values in the same order as the config
        ordered_lines = [self.lines[i] for i in self._line_order]
        line_names = [line.name or f'line_{i}' for i, line in enumerate(ordered_lines)]

        # Flux units: both injected and recovered are in 1e-20 erg/s/cm2
        flux_injected = np.array([line.flux for line in ordered_lines])
        fwhm_injected = np.array([line.fwhm_kms for line in ordered_lines])
        z_injected = np.full(len(ordered_lines), self.base_spectra.redshift_initial)

        # Compute SNR for each line (flux / uncertainty)
        snr = flux_recovered / np.maximum(flux_sigma, 1e-10)

        # Calculate absolute errors
        flux_abs_error = np.abs(flux_recovered - flux_injected)
        fwhm_abs_error = np.abs(fwhm_recovered - fwhm_injected)
        z_abs_error = np.abs(z_recovered - z_injected)

        # Calculate fractional errors (%)
        flux_pct_error = flux_abs_error / flux_injected
        fwhm_pct_error = fwhm_abs_error / fwhm_injected

        # Calculate error in units of sigma
        flux_nsigma = flux_abs_error / np.maximum(flux_sigma, 1e-10)
        fwhm_nsigma = fwhm_abs_error / np.maximum(fwhm_sigma, 1e-10)
        z_nsigma = z_abs_error / np.maximum(z_sigma, 1e-10)

        # Build base metrics
        metrics = {
            'flux_mean_error': float(np.mean(flux_pct_error)),
            'fwhm_mean_error': float(np.mean(fwhm_pct_error)),
            'z_mean_error': float(np.mean(z_abs_error)),
            'flux_max_nsigma': float(np.max(flux_nsigma)),
            'fwhm_max_nsigma': float(np.max(fwhm_nsigma)),
            'z_max_nsigma': float(np.max(z_nsigma)),
        }

        # Check if within sigma tolerance (only for line parameters)
        flux_pass = np.all(flux_nsigma < sigma_tolerance)
        fwhm_pass = np.all(fwhm_nsigma < sigma_tolerance)
        z_pass = np.all(z_nsigma < sigma_tolerance)
        cont_pass = True  # Default to True unless validate_continuum is enabled

        # Extract continuum parameters if available (for reporting, not validation by default)
        if self.injected_continuum is not None and 'cont_angle' in self.samples:
            cont_angle_all = self.samples['cont_angle']
            cont_offset_all = self.samples['cont_offset']

            # Ensure arrays are at least 1D (median/percentile can return scalars for single-element arrays)
            cont_angle_recovered = np.atleast_1d(np.median(cont_angle_all, axis=0))
            cont_offset_recovered = np.atleast_1d(np.median(cont_offset_all, axis=0))

            cont_angle_lo = np.atleast_1d(np.percentile(cont_angle_all, 16, axis=0))
            cont_angle_hi = np.atleast_1d(np.percentile(cont_angle_all, 84, axis=0))
            cont_offset_lo = np.atleast_1d(np.percentile(cont_offset_all, 16, axis=0))
            cont_offset_hi = np.atleast_1d(np.percentile(cont_offset_all, 84, axis=0))

            cont_angle_sigma = (cont_angle_hi - cont_angle_lo) / 2
            cont_offset_sigma = (cont_offset_hi - cont_offset_lo) / 2

            cont_angle_injected = np.atleast_1d(self.injected_continuum.angles)
            cont_offset_injected = np.atleast_1d(self.injected_continuum.offsets)

            cont_angle_abs_error = np.abs(cont_angle_recovered - cont_angle_injected)
            cont_offset_abs_error = np.abs(cont_offset_recovered - cont_offset_injected)

            cont_angle_nsigma = cont_angle_abs_error / np.maximum(cont_angle_sigma, 1e-10)
            cont_offset_nsigma = cont_offset_abs_error / np.maximum(cont_offset_sigma, 1e-10)

            # Only check continuum for pass/fail if validate_continuum=True
            if validate_continuum:
                cont_pass = np.all(cont_angle_nsigma < sigma_tolerance) and np.all(
                    cont_offset_nsigma < sigma_tolerance
                )

            # Add continuum metrics (always reported for information)
            metrics['cont_angle_max_nsigma'] = float(np.max(cont_angle_nsigma))
            metrics['cont_offset_max_nsigma'] = float(np.max(cont_offset_nsigma))

        # Pass/fail based on line parameters only (unless validate_continuum=True)
        passed = flux_pass and fwhm_pass and z_pass and cont_pass

        # Build details string
        details_parts = []
        for i, line in enumerate(ordered_lines):
            name = line.name or f'line_{i}'
            details_parts.append(
                f'{name}: '
                f'flux={flux_nsigma[i]:.1f}σ ({"PASS" if flux_nsigma[i] < sigma_tolerance else "FAIL"}), '
                f'fwhm={fwhm_nsigma[i]:.1f}σ ({"PASS" if fwhm_nsigma[i] < sigma_tolerance else "FAIL"}), '
                f'z={z_nsigma[i]:.1f}σ ({"PASS" if z_nsigma[i] < sigma_tolerance else "FAIL"})'
            )

        # Build injected/recovered/uncertainties/nsigma dicts
        injected_dict = {'flux': flux_injected, 'fwhm': fwhm_injected, 'redshift': z_injected}
        recovered_dict = {'flux': flux_recovered, 'fwhm': fwhm_recovered, 'redshift': z_recovered}
        uncertainties_dict = {
            'flux_lo': flux_lo,
            'flux_hi': flux_hi,
            'flux_sigma': flux_sigma,
            'fwhm_lo': fwhm_lo,
            'fwhm_hi': fwhm_hi,
            'fwhm_sigma': fwhm_sigma,
            'z_lo': z_lo,
            'z_hi': z_hi,
            'z_sigma': z_sigma,
        }
        nsigma_dict = {'flux': flux_nsigma, 'fwhm': fwhm_nsigma, 'redshift': z_nsigma}

        # Add continuum to dicts if available
        if 'cont_angle_max_nsigma' in metrics:
            # Continuum was processed, add to details
            details_parts.append('\nContinuum:')
            for i in range(len(cont_angle_nsigma)):
                details_parts.append(
                    f'  Region {i}: '
                    f'angle={cont_angle_nsigma[i]:.1f}σ ({"PASS" if cont_angle_nsigma[i] < sigma_tolerance else "FAIL"}), '
                    f'offset={cont_offset_nsigma[i]:.1f}σ ({"PASS" if cont_offset_nsigma[i] < sigma_tolerance else "FAIL"})'
                )

            # Add to result dicts
            injected_dict['cont_angle'] = cont_angle_injected
            injected_dict['cont_offset'] = cont_offset_injected
            recovered_dict['cont_angle'] = cont_angle_recovered
            recovered_dict['cont_offset'] = cont_offset_recovered
            uncertainties_dict['cont_angle_sigma'] = cont_angle_sigma
            uncertainties_dict['cont_offset_sigma'] = cont_offset_sigma
            nsigma_dict['cont_angle'] = cont_angle_nsigma
            nsigma_dict['cont_offset'] = cont_offset_nsigma

        details = '\n'.join(details_parts)

        return ValidationResult(
            passed=passed,
            metrics=metrics,
            injected=injected_dict,
            recovered=recovered_dict,
            uncertainties=uncertainties_dict,
            nsigma=nsigma_dict,
            snr=snr,
            line_names=line_names,
            sigma_tolerance=sigma_tolerance,
            details=details,
        )

    def plot_results(self, **plot_kwargs):
        """Plot the fitting results.

        Parameters
        ----------
        **plot_kwargs
            Additional arguments passed to plotResults.

        Returns
        -------
        tuple
            (fig, data) from plotResults.
        """
        from unite.plotting import plotResults

        if self.output_dir is None:
            raise ValueError('Must run fit() before plot_results()')

        return plotResults(
            self.config,
            rows=self.rows,
            output_dir=str(self.output_dir),
            spectra=self.injected_spectra,
            plot_kwargs=plot_kwargs,
        )


def inject_synthetic_lines(
    inspec: NIRSpecSpectrum,
    lines: List[SyntheticLine],
    continuum: SyntheticContinuum | float | None = None,
    rng: np.random.Generator | None = None,
    lsf_scale: float = 1.2,
) -> NIRSpecSpectrum:
    """Inject synthetic lines into a spectrum using its actual LSF.

    This function uses the spectrum's built-in LSF (from calibration files)
    to properly convolve the injected lines.

    Parameters
    ----------
    inspec : NIRSpecSpectrum
        Input spectrum (will be deep copied).
    lines : List[SyntheticLine]
        Lines to inject.
    continuum : SyntheticContinuum or float, optional
        Continuum model. If float, uses flat continuum. If None, uses median of existing flux.
    rng : numpy.random.Generator, optional
        Random generator for noise. Default: seed=0.
    lsf_scale : float
        LSF scaling factor passed to spec.lsf().

    Returns
    -------
    NIRSpecSpectrum
        Modified spectrum with injected lines.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    spec = deepcopy(inspec)
    low, wave, high, _flux, err = spec()

    # Evaluate continuum model
    if continuum is None:
        continuum = SyntheticContinuum.from_spectrum_median(spec, rng=rng)
    elif isinstance(continuum, (int, float)):
        # Flat continuum for backward compatibility
        wave_range = wave.max() - wave.min()
        n_regions = max(1, int(wave_range / 0.5))
        edges = np.linspace(wave.min(), wave.max(), n_regions + 1)
        regions = np.column_stack([edges[:-1], edges[1:]])
        angles = np.zeros(n_regions)
        offsets = np.full(n_regions, float(continuum))
        continuum = SyntheticContinuum(regions=regions, angles=angles, offsets=offsets)

    # Evaluate continuum at wavelengths using optimized.linearContinua
    cont_centers = jnp.array(continuum.regions.mean(axis=1))
    cont_model = optimized.linearContinua(
        jnp.array(wave),
        cont_centers,
        jnp.array(continuum.angles),
        jnp.array(continuum.offsets),
        jnp.array(continuum.regions),
    ).sum(1)
    cont_flux = np.asarray(cont_model)

    # Redshift factor
    opz = 1.0 + spec.redshift_initial

    n_lines = len(lines)
    if n_lines == 0:
        spec.flux = cont_flux + rng.normal(0.0, err)
        return spec

    # Rest wavelengths in Angstrom -> spectrum wavelength units (micron)
    rest_wavelengths = np.array([line.wavelength for line in lines])
    centers = rest_wavelengths * u.AA.to(spec.λ_unit) * opz

    # FWHMs in km/s -> wavelength units
    fwhm_kms = np.array([line.fwhm_kms for line in lines])
    fwhm = centers * fwhm_kms / C

    # Line fluxes: convert from 1e-20 erg/s/cm^2 to internal units
    # Internal units: fλ_unit * λ_unit = 1e-20 erg/(s cm2 AA) * micron
    line_fluxes = np.array([line.flux for line in lines])
    line_fluxes = line_fluxes / spec.λ_unit.to(u.AA)

    # Profile type indices
    profile_map = {
        'gaussian': optimized.GAUSSIAN,
        'lorentzian': optimized.LORENTZIAN,
        'exponential': optimized.EXPONENTIAL,
    }
    type_idx = jnp.array([profile_map.get(line.profile, optimized.GAUSSIAN) for line in lines])

    # Get LSF at line centers using the spectrum's actual LSF function
    lsf = spec.lsf(centers, lsf_scale)

    # Convert to JAX arrays
    low_j = jnp.asarray(low)
    high_j = jnp.asarray(high)
    centers_j = jnp.asarray(centers)
    fwhm_j = jnp.asarray(fwhm)
    lsf_j = jnp.asarray(lsf)
    line_fluxes_j = jnp.asarray(line_fluxes)

    # Log injection details
    for i, line in enumerate(lines):
        lsf_R = centers[i] / lsf[i]
        logger.info(
            f'Injecting {line.name or f"line_{i}"}: '
            f'λ={centers[i]:.3f} {spec.λ_unit}, flux={line.flux:.2f}, '
            f'FWHM={fwhm_kms[i]:.0f} km/s, LSF R={lsf_R:.0f}'
        )

    # Integrate line profiles over pixel bins
    pixints = optimized.integrate(low_j, high_j, centers_j, lsf_j, fwhm_j, type_idx).T

    # Convert to flux density (divide by bin width)
    f_lambda = pixints / (high_j - low_j)[:, jnp.newaxis]

    # Multiply by line fluxes
    line_model = f_lambda * line_fluxes_j

    # Handle absorption lines
    absorption_mask = np.array([line.is_absorption for line in lines])
    if absorption_mask.any():
        signs = jnp.where(jnp.array(absorption_mask), -1.0, 1.0)
        line_model = line_model * signs

    # Sum all line contributions
    total_line_flux = np.asarray(line_model.sum(axis=1))

    # Build final model: continuum + lines + noise
    model_flux = cont_flux + total_line_flux
    noise = rng.normal(0.0, err)
    spec.flux = model_flux + noise

    return spec
