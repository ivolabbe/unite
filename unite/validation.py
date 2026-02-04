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
    """Configuration for synthetic continuum.

    For linear: uses regions, angles, offsets.
    For blackbody: uses bb_amplitude, bb_temperature.
    For modified_blackbody: uses bb_amplitude, bb_temperature, bb_beta.
    For attenuated_blackbody: uses bb_amplitude, bb_temperature, bb_tau_v, bb_alpha.

    Attributes
    ----------
    regions : np.ndarray
        Continuum regions (Nc, 2) array of [low, high] wavelength bounds in microns.
        For linear continuum only.
    angles : np.ndarray
        Angles for each continuum segment (Nc,) in radians. For linear continuum only.
    offsets : np.ndarray
        Offset heights for each continuum segment (Nc,) in flux units. For linear continuum only.
    continuum_type : str
        Type of continuum: 'linear', 'blackbody', 'modified_blackbody', or 'attenuated_blackbody'.
    bb_amplitude : float, optional
        Amplitude for blackbody continuum.
    bb_temperature : float, optional
        Temperature in Kelvin for blackbody continuum.
    bb_beta : float, optional
        Emissivity index for modified blackbody.
    bb_tau_v : float, optional
        V-band optical depth for attenuated blackbody.
    bb_alpha : float, optional
        Attenuation power-law slope for attenuated blackbody.
    pivot_micron : float
        Pivot wavelength for normalization (microns).
    """

    regions: np.ndarray
    angles: np.ndarray
    offsets: np.ndarray
    continuum_type: str = 'linear'
    bb_amplitude: float | None = None
    bb_temperature: float | None = None
    bb_beta: float | None = None
    bb_tau_v: float | None = None
    bb_alpha: float | None = None
    pivot_micron: float = 1.0
    temp_type: str = 'default'

    @classmethod
    def from_params(
        cls,
        type: str = 'linear',
        temperature: float | None = None,
        normalization: float | None = None,
        beta: float | None = None,
        tau_v: float | None = None,
        alpha: float | None = None,
        pivot_micron: float = 1.0,
        temp_type: str = 'default',
        regions: np.ndarray | None = None,
        angles: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
    ) -> 'SyntheticContinuum':
        """
        Create continuum from simplified parameters.

        For blackbody/modified blackbody/attenuated blackbody, normalization is the observed-frame f_lambda
        flux density at the pivot wavelength in units of 1e-20 erg/s/cm²/Å (JWST standard).

        Parameters
        ----------
        type : str
            Continuum type: 'linear', 'blackbody', 'modified_blackbody', or 'attenuated_blackbody'
        temperature : float, optional
            Temperature in Kelvin (for BB/MBB/ABB)
        normalization : float, optional
            Observed-frame f_lambda at pivot wavelength in units of 1e-20 erg/s/cm²/Å
            Example: normalization=10 means f_lambda = 10 * 1e-20 = 1e-19 erg/s/cm²/Å
        beta : float, optional
            Emissivity index (for MBB only)
        tau_v : float, optional
            V-band optical depth (for ABB only)
        alpha : float, optional
            Attenuation power-law slope (for ABB only). Range: -0.4 (MW) to -2.0 (very steep)
        pivot_micron : float
            Pivot wavelength for normalization in microns (default: 1.0)
        temp_type : str
            Temperature prior type: 'hot' (20k-100k), 'warm' (2k-15k), 'dust' (20-1500), 'default' (1k-30k)
        regions : np.ndarray, optional
            Continuum regions for linear model
        angles : np.ndarray, optional
            Angles for linear model
        offsets : np.ndarray, optional
            Offsets for linear model in units of 1e-20 erg/s/cm²/Å

        Returns
        -------
        SyntheticContinuum
            Configured continuum model

        Examples
        --------
        >>> # Blackbody continuum: f_lambda = 10 * 1e-20 at pivot
        >>> cont = SyntheticContinuum.from_params(
        ...     type='blackbody',
        ...     temperature=5000,
        ...     normalization=10,        # in units of 1e-20 erg/s/cm²/Å
        ...     pivot_micron=1.0
        ... )
        >>> # Modified blackbody continuum
        >>> cont = SyntheticContinuum.from_params(
        ...     type='modified_blackbody',
        ...     temperature=5000,
        ...     normalization=10,        # in units of 1e-20 erg/s/cm²/Å
        ...     beta=1.5
        ... )
        >>> # Attenuated blackbody continuum
        >>> cont = SyntheticContinuum.from_params(
        ...     type='attenuated_blackbody',
        ...     temperature=5000,
        ...     normalization=10,        # in units of 1e-20 erg/s/cm²/Å
        ...     tau_v=1.5,
        ...     alpha=-0.7               # LMC-like attenuation slope
        ... )
        """
        type_lower = type.lower()

        if type_lower in ('blackbody', 'modified_blackbody', 'attenuated_blackbody'):
            if temperature is None or normalization is None:
                raise ValueError(f'{type} continuum requires temperature and normalization parameters')

            #  from units of 1e-20 erg/s/cm²/Å
            amplitude = normalization

            continuum_type = type_lower
            return cls(
                regions=np.array([[0, 10]]),  # Dummy regions (unused for BB)
                angles=np.array([0.0]),
                offsets=np.array([amplitude]),
                continuum_type=continuum_type,
                bb_amplitude=amplitude,
                bb_temperature=temperature,
                bb_beta=beta,
                bb_tau_v=tau_v,
                bb_alpha=alpha,
                pivot_micron=pivot_micron,
                temp_type=temp_type,
            )

        elif type_lower == 'linear':
            if regions is None or angles is None or offsets is None:
                raise ValueError('linear continuum requires regions, angles, and offsets')

            #  offsets from units of 1e-20 erg/s/cm²/Å
            offsets_scaled = offsets

            return cls(
                regions=regions,
                angles=angles,
                offsets=offsets_scaled,
                continuum_type='linear',
                pivot_micron=PivotMicron,
            )

        else:
            raise ValueError(f'Unknown continuum type: {type}')

    @classmethod
    def from_blackbody(
        cls,
        amplitude: float,
        temperature: float,
        pivot_micron: float = 1.0,
        beta: float | None = None,
        temp_type: str = 'default',
    ) -> 'SyntheticContinuum':
        """
        Create blackbody or modified blackbody continuum for validation.

        DEPRECATED: Use from_params() instead for a cleaner API.

        Parameters
        ----------
        amplitude : float
            Blackbody amplitude
        temperature : float
            Temperature in Kelvin
        pivot_micron : float
            Pivot wavelength for normalization (microns)
        beta : float, optional
            If provided, creates modified blackbody with this emissivity index
        temp_type : str
            Temperature prior type: 'hot', 'warm', 'dust', 'default'

        Returns
        -------
        SyntheticContinuum
            Blackbody continuum model
        """
        continuum_type = 'modified_blackbody' if beta is not None else 'blackbody'
        return cls(
            regions=np.array([[0, 10]]),  # Dummy regions (unused for BB)
            angles=np.array([0.0]),
            offsets=np.array([amplitude]),
            continuum_type=continuum_type,
            bb_amplitude=amplitude,
            bb_temperature=temperature,
            bb_beta=beta,
            pivot_micron=pivot_micron,
            temp_type=temp_type,
        )

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

        # Add BB/MBB continuum table if available
        if 'bb_amplitude' in self.injected:
            # Determine continuum model type (check for beta to distinguish MBB from BB)
            if 'bb_beta' in self.injected:
                cont_type = 'Modified Blackbody Continuum'
            else:
                cont_type = 'Blackbody Continuum'

            lines.append(f'\n{cont_type}:')
            lines.append(f'{"Parameter":<12} {"Injected":>12} {"Recovered":>12} {"[16%, 84%]":>22}')
            lines.append('-' * 80)

            # Amplitude
            amp_inj = self.injected['bb_amplitude']
            amp_rec = self.recovered['bb_amplitude']
            amp_lo = self.uncertainties['bb_amplitude_lo']
            amp_hi = self.uncertainties['bb_amplitude_hi']
            lines.append(
                f'{"amplitude":<12} {amp_inj:>12.2e} {amp_rec:>12.2e} '
                f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
            )

            # Temperature
            temp_inj = self.injected['bb_temperature']
            temp_rec = self.recovered['bb_temperature']
            temp_lo = self.uncertainties['bb_temperature_lo']
            temp_hi = self.uncertainties['bb_temperature_hi']
            lines.append(
                f'{"temperature":<12} {temp_inj:>12.0f} {temp_rec:>12.0f} '
                f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
            )

            # Beta (for modified blackbody only)
            if 'bb_beta' in self.injected:
                beta_inj = self.injected['bb_beta']
                beta_rec = self.recovered['bb_beta']
                beta_lo = self.uncertainties['bb_beta_lo']
                beta_hi = self.uncertainties['bb_beta_hi']
                lines.append(
                    f'{"beta":<12} {beta_inj:>12.2f} {beta_rec:>12.2f} '
                    f'[{beta_lo:>9.2f}, {beta_hi:>9.2f}]'
                )

            # Tau_V (for attenuated blackbody only)
            if 'bb_tau_v' in self.injected:
                tau_v_inj = self.injected['bb_tau_v']
                tau_v_rec = self.recovered['bb_tau_v']
                tau_v_lo = self.uncertainties['bb_tau_v_lo']
                tau_v_hi = self.uncertainties['bb_tau_v_hi']
                lines.append(
                    f'{"tau_V":<12} {tau_v_inj:>12.2f} {tau_v_rec:>12.2f} '
                    f'[{tau_v_lo:>9.2f}, {tau_v_hi:>9.2f}]'
                )

            # Alpha (for attenuated blackbody only)
            if 'bb_alpha' in self.injected:
                alpha_inj = self.injected['bb_alpha']
                alpha_rec = self.recovered['bb_alpha']
                alpha_lo = self.uncertainties['bb_alpha_lo']
                alpha_hi = self.uncertainties['bb_alpha_hi']
                lines.append(
                    f'{"alpha":<12} {alpha_inj:>12.2f} {alpha_rec:>12.2f} '
                    f'[{alpha_lo:>9.2f}, {alpha_hi:>9.2f}]'
                )

            lines.append('-' * 80)

        # Add composite continuum table if available
        if self.recovered.get('continuum_type') == 'composite':
            # Check if we have injected values
            has_injected = any(k.endswith('_amplitude') for k in self.injected.keys() if k.startswith(('mbb', 'abb', 'bb')))

            if has_injected:
                lines.append(f'\nComposite Continuum:')
                lines.append(f'{"Component":<12} {"Parameter":<12} {"Injected":>12} {"Recovered":>12} {"[16%, 84%]":>22}')
            else:
                lines.append(f'\nComposite Continuum (fitted parameters):')
                lines.append(f'{"Component":<12} {"Parameter":<12} {"Recovered":>12} {"[16%, 84%]":>22}')
            lines.append('-' * 80)

            # Display each component (check for MBB, ABB, and BB)
            for i in range(1, 10):  # Support up to 9 components
                # Check for MBB component (modified blackbody with beta)
                mbb_amp_key = f'mbb{i}_amplitude'
                if mbb_amp_key in self.recovered:
                    lines.append(f'MBB {i}:')

                    # Amplitude
                    amp = self.recovered[mbb_amp_key]
                    amp_lo = self.recovered[f'mbb{i}_amplitude_lo']
                    amp_hi = self.recovered[f'mbb{i}_amplitude_hi']
                    if has_injected and mbb_amp_key in self.injected:
                        amp_inj = self.injected[mbb_amp_key]
                        lines.append(
                            f'{"":<12} {"amplitude":<12} {amp_inj:>12.2e} {amp:>12.2e} '
                            f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"amplitude":<12} {amp:>12.2e} '
                            f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
                        )

                    # Temperature
                    temp = self.recovered[f'mbb{i}_temperature']
                    temp_lo = self.recovered[f'mbb{i}_temperature_lo']
                    temp_hi = self.recovered[f'mbb{i}_temperature_hi']
                    if has_injected and f'mbb{i}_temperature' in self.injected:
                        temp_inj = self.injected[f'mbb{i}_temperature']
                        lines.append(
                            f'{"":<12} {"temperature":<12} {temp_inj:>12.0f} {temp:>12.0f} '
                            f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"temperature":<12} {temp:>12.0f} '
                            f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
                    )

                    # Beta
                    beta = self.recovered[f'mbb{i}_beta']
                    beta_lo = self.recovered[f'mbb{i}_beta_lo']
                    beta_hi = self.recovered[f'mbb{i}_beta_hi']
                    if has_injected and f'mbb{i}_beta' in self.injected:
                        beta_inj = self.injected[f'mbb{i}_beta']
                        lines.append(
                            f'{"":<12} {"beta":<12} {beta_inj:>12.2f} {beta:>12.2f} '
                            f'[{beta_lo:>9.2f}, {beta_hi:>9.2f}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"beta":<12} {beta:>12.2f} '
                            f'[{beta_lo:>9.2f}, {beta_hi:>9.2f}]'
                        )
                    lines.append('')  # Blank line between components

                # Check for ABB component (attenuated blackbody with tau_v)
                abb_amp_key = f'abb{i}_amplitude'
                if abb_amp_key in self.recovered:
                    lines.append(f'ABB {i}:')

                    # Amplitude
                    amp = self.recovered[abb_amp_key]
                    amp_lo = self.recovered[f'abb{i}_amplitude_lo']
                    amp_hi = self.recovered[f'abb{i}_amplitude_hi']
                    if has_injected and abb_amp_key in self.injected:
                        amp_inj = self.injected[abb_amp_key]
                        lines.append(
                            f'{"":<12} {"amplitude":<12} {amp_inj:>12.2e} {amp:>12.2e} '
                            f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"amplitude":<12} {amp:>12.2e} '
                            f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
                        )

                    # Temperature
                    temp = self.recovered[f'abb{i}_temperature']
                    temp_lo = self.recovered[f'abb{i}_temperature_lo']
                    temp_hi = self.recovered[f'abb{i}_temperature_hi']
                    if has_injected and f'abb{i}_temperature' in self.injected:
                        temp_inj = self.injected[f'abb{i}_temperature']
                        lines.append(
                            f'{"":<12} {"temperature":<12} {temp_inj:>12.0f} {temp:>12.0f} '
                            f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"temperature":<12} {temp:>12.0f} '
                            f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
                    )

                    # Tau_V
                    tau_v = self.recovered[f'abb{i}_tau_v']
                    tau_v_lo = self.recovered[f'abb{i}_tau_v_lo']
                    tau_v_hi = self.recovered[f'abb{i}_tau_v_hi']
                    if has_injected and f'abb{i}_tau_v' in self.injected:
                        tau_v_inj = self.injected[f'abb{i}_tau_v']
                        lines.append(
                            f'{"":<12} {"tau_V":<12} {tau_v_inj:>12.2f} {tau_v:>12.2f} '
                            f'[{tau_v_lo:>9.2f}, {tau_v_hi:>9.2f}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"tau_V":<12} {tau_v:>12.2f} '
                            f'[{tau_v_lo:>9.2f}, {tau_v_hi:>9.2f}]'
                        )

                    # Alpha
                    alpha = self.recovered[f'abb{i}_alpha']
                    alpha_lo = self.recovered[f'abb{i}_alpha_lo']
                    alpha_hi = self.recovered[f'abb{i}_alpha_hi']
                    if has_injected and f'abb{i}_alpha' in self.injected:
                        alpha_inj = self.injected[f'abb{i}_alpha']
                        lines.append(
                            f'{"":<12} {"alpha":<12} {alpha_inj:>12.2f} {alpha:>12.2f} '
                            f'[{alpha_lo:>9.2f}, {alpha_hi:>9.2f}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"alpha":<12} {alpha:>12.2f} '
                            f'[{alpha_lo:>9.2f}, {alpha_hi:>9.2f}]'
                        )
                    lines.append('')  # Blank line between components

                # Check for plain BB component (blackbody, uncommon but possible)
                bb_amp_key = f'bb{i}_amplitude'
                if bb_amp_key in self.recovered:
                    lines.append(f'BB {i}:')

                    # Amplitude
                    amp = self.recovered[bb_amp_key]
                    amp_lo = self.recovered[f'bb{i}_amplitude_lo']
                    amp_hi = self.recovered[f'bb{i}_amplitude_hi']
                    if has_injected and bb_amp_key in self.injected:
                        amp_inj = self.injected[bb_amp_key]
                        lines.append(
                            f'{"":<12} {"amplitude":<12} {amp_inj:>12.2e} {amp:>12.2e} '
                            f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"amplitude":<12} {amp:>12.2e} '
                            f'[{amp_lo:>9.2e}, {amp_hi:>9.2e}]'
                        )

                    # Temperature
                    temp = self.recovered[f'bb{i}_temperature']
                    temp_lo = self.recovered[f'bb{i}_temperature_lo']
                    temp_hi = self.recovered[f'bb{i}_temperature_hi']
                    if has_injected and f'bb{i}_temperature' in self.injected:
                        temp_inj = self.injected[f'bb{i}_temperature']
                        lines.append(
                            f'{"":<12} {"temperature":<12} {temp_inj:>12.0f} {temp:>12.0f} '
                            f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
                        )
                    else:
                        lines.append(
                            f'{"":<12} {"temperature":<12} {temp:>12.0f} '
                            f'[{temp_lo:>9.0f}, {temp_hi:>9.0f}]'
                    )
                    lines.append('')  # Blank line between components

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

        # Add BB/MBB continuum summary if available
        if 'bb_amplitude_nsigma' in self.metrics:
            # Check if it's MBB (has beta) or pure BB
            if 'bb_beta_nsigma' in self.metrics:
                prefix = 'MBB'
            else:
                prefix = 'BB'

            lines.append(
                f'  {prefix} amplitude:    ' f'offset = {self.metrics["bb_amplitude_nsigma"]:.1f}σ'
            )
            lines.append(
                f'  {prefix} temperature:  ' f'offset = {self.metrics["bb_temperature_nsigma"]:.1f}σ'
            )

            if 'bb_beta_nsigma' in self.metrics:
                lines.append(
                    f'  {prefix} beta:         ' f'offset = {self.metrics["bb_beta_nsigma"]:.1f}σ'
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
        rows: Table | None = None,
        lines: List[SyntheticLine] | None = None,
        continuum: SyntheticContinuum | List[SyntheticContinuum] | float | None = None,
        rng_seed: int = 0,
        spectrum_path: str | None = None,
        continuum_only: bool = False,
        name: str = 'validation',
        redshift: float | None = None,
    ) -> None:
        """Initialize ValidationSuite.

        Parameters
        ----------
        rows : Table, optional
            Astropy table with spectrum metadata. Required if spectrum_path not provided.
        lines : List[SyntheticLine], optional
            Lines to inject. Required unless continuum_only=True.
        continuum : SyntheticContinuum or List or float, optional
            Continuum model(s).
        rng_seed : int
            Random seed.
        spectrum_path : str, optional
            Path to spectrum file. If provided, creates rows table automatically.
        continuum_only : bool
            If True, fit only continuum (no emission lines).
        name : str
            Config name for output files.
        redshift : float, optional
            Redshift of the source. Only used when spectrum_path is provided.
            If not specified, defaults to 2.0.
        """
        # Handle spectrum_path convenience parameter
        if spectrum_path is not None:
            if rows is not None:
                raise ValueError('Provide either rows or spectrum_path, not both')
            # Create rows table from spectrum path
            from astropy.table import Table as ATable
            from pathlib import Path as PathLib

            spec_path = PathLib(spectrum_path)

            # Infer grating from filename
            grating = 'PRISM' if 'prism' in spec_path.name.lower() else 'G235M'

            # Use provided redshift or default to 2.0
            z_value = redshift if redshift is not None else 2.0

            rows = ATable({
                'root': ['validation'],
                'srcid': [0],
                'file': [spec_path.name],
                'spectra_directory': [str(spec_path.parent.resolve())],
                'grade': [1],  # Required by NIRSpecSpectra
                'grating': [grating],  # Required by NIRSpecSpectra
                'z': [z_value],
                'zfit': [z_value],
            })

        if rows is None:
            raise ValueError('Must provide either rows or spectrum_path')

        # Check for continuum-only mode
        if continuum_only:
            if lines is not None:
                raise ValueError('continuum_only=True requires lines=None')
        else:
            if lines is None:
                raise ValueError('lines is required unless continuum_only=True')

        self.rows = rows
        self.lines = lines if lines is not None else []
        self.rng = np.random.default_rng(rng_seed)
        self.continuum_only = continuum_only
        self.name = name

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
        self.injected_continuum: SyntheticContinuum | List[SyntheticContinuum] | None = None
        self.config: dict | None = None
        self.samples: dict | None = None
        self.output_dir: Path | None = None
        self._line_order: List[int] | None = None
        self.fit_result: 'FitResults | None' = None

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
        injected_spec = inject_synthetic_lines(
            spec, self.lines, continuum=continuum, rng=self.rng, lsf_scale=lsf_scale
        )

        # Save mock spectrum to disk in spectra_mock/ directory
        from astropy.io import fits
        import os

        # Determine mock directory - use spectra_directory column if present
        if 'spectra_directory' in self.rows.colnames:
            # Get base directory and create mock version
            orig_spec_dir = str(self.rows[spectrum_idx]['spectra_directory'])
            # Replace 'spectra' with 'spectra_mock'
            if 'spectra' in os.path.basename(orig_spec_dir):
                mock_dir = os.path.join(os.path.dirname(orig_spec_dir), 'spectra_mock')
            else:
                mock_dir = orig_spec_dir + '_mock'
            has_spec_dir_column = True
        else:
            # No spectra_directory column - use spectra_mock in cwd
            mock_dir = 'spectra_mock'
            has_spec_dir_column = False

        os.makedirs(mock_dir, exist_ok=True)

        # Keep the same filename
        orig_file = str(self.rows[spectrum_idx]['file'])
        filename = os.path.basename(orig_file)

        # Full path for saving
        mock_path = os.path.join(mock_dir, filename)

        # Save to FITS - match original spectrum format (SPEC1D extension with uJy units)
        from astropy.table import Table as AstropyTable, MaskedColumn
        from astropy import units as u_astropy

        # Convert flux from 1e-20 erg/s/cm2/A to uJy
        wave_um = injected_spec.wave * u_astropy.um
        flux_flam = injected_spec.flux * (1e-20 * u_astropy.erg / u_astropy.s / u_astropy.cm**2 / u_astropy.AA)
        err_flam = injected_spec.err * (1e-20 * u_astropy.erg / u_astropy.s / u_astropy.cm**2 / u_astropy.AA)

        flux_fnu = flux_flam.to(u_astropy.uJy, equivalencies=u_astropy.spectral_density(wave_um))
        err_fnu = err_flam.to(u_astropy.uJy, equivalencies=u_astropy.spectral_density(wave_um))

        # Create table with masked columns (matches original format)
        n_pix = len(injected_spec.wave)
        spec_table = AstropyTable()
        spec_table['wave'] = injected_spec.wave * u_astropy.um
        spec_table['flux'] = MaskedColumn(flux_fnu.value, mask=np.zeros(n_pix, dtype=bool), unit=u_astropy.uJy)
        spec_table['err'] = MaskedColumn(err_fnu.value, mask=np.zeros(n_pix, dtype=bool), unit=u_astropy.uJy)
        spec_table['sky'] = MaskedColumn(np.zeros(n_pix), mask=np.zeros(n_pix, dtype=bool), unit=u_astropy.uJy)
        spec_table['path_corr'] = np.ones(n_pix)
        spec_table['npix'] = np.ones(n_pix)

        # Convert to FITS BinTableHDU and create HDUList
        hdu_primary = fits.PrimaryHDU()
        hdu_primary.header['EXTNAME'] = 'PRIMARY'

        # Convert table to FITS BinTableHDU
        hdu_spec1d = fits.table_to_hdu(spec_table)
        hdu_spec1d.header['EXTNAME'] = 'SPEC1D'

        hdul = fits.HDUList([hdu_primary, hdu_spec1d])
        hdul.writeto(mock_path, overwrite=True)
        logger.info(f'Saved mock spectrum to {mock_path}')

        # Update rows table to point to mock directory
        from astropy.table import Table as ATable, Column

        self.rows = ATable(self.rows, copy=True)

        # Update or add spectra_directory column
        # NOTE: Must recreate column to avoid string truncation issues
        if has_spec_dir_column:
            # Remove old column and create new one with updated data
            spec_dir_data = [str(row['spectra_directory']) for row in self.rows]
            spec_dir_data[spectrum_idx] = mock_dir
            self.rows.remove_column('spectra_directory')
            spec_dir_col = Column(name='spectra_directory', data=spec_dir_data)
            self.rows.add_column(spec_dir_col)
        else:
            # Add spectra_directory column for all rows
            spec_dir_data = [mock_dir] * len(self.rows)
            spec_dir_col = Column(name='spectra_directory', data=spec_dir_data)
            self.rows.add_column(spec_dir_col)

        # file column stays the same (just the basename)

        # Reload spectra from disk (from mock directory)
        spectra = NIRSpecSpectra(self.rows)

        self.injected_spectra = spectra
        return spectra

    def generate_config(self) -> dict:
        """Generate UNITE config from synthetic lines.

        Returns
        -------
        dict
            UNITE configuration dictionary.
        """
        # Build config structure
        config: dict = {'Name': self.name, 'Unit': 'AA', 'Groups': {}}

        # Continuum-only mode: empty Groups, add mask_lines and continuum_only flag
        if self.continuum_only:
            config['continuum_only'] = True
            config['mask_lines'] = 'default'  # Use DEFAULT_MASK_LINES from defaults.py

            # Add continuum configuration
            if self.continuum is not None:
                if isinstance(self.continuum, list):
                    config['continuum'] = []
                    for cont_component in self.continuum:
                        if cont_component.continuum_type == 'blackbody':
                            config['continuum'].append({
                                'type': 'blackbody',
                                'pivot_micron': cont_component.pivot_micron,
                                'temp_type': cont_component.temp_type,
                            })
                        elif cont_component.continuum_type == 'modified_blackbody':
                            config['continuum'].append({
                                'type': 'modified_blackbody',
                                'pivot_micron': cont_component.pivot_micron,
                                'temp_type': cont_component.temp_type,
                            })
                elif self.continuum.continuum_type != 'linear':
                    if self.continuum.continuum_type == 'blackbody':
                        config['continuum'] = {
                            'type': 'blackbody',
                            'pivot_micron': self.continuum.pivot_micron,
                            'temp_type': self.continuum.temp_type,
                        }
                    elif self.continuum.continuum_type == 'modified_blackbody':
                        config['continuum'] = {
                            'type': 'modified_blackbody',
                            'pivot_micron': self.continuum.pivot_micron,
                            'temp_type': self.continuum.temp_type,
                        }

            self._line_order = []
            self.config = config
            return config

        # Standard mode: generate line groups
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
            elif line.fwhm_kms < 700:
                line_type = 'narrow'
            elif line.fwhm_kms > 1000:
                line_type = 'broad'
            else:
                # 700 <= FWHM <= 1000: intermediate, default to broad
                line_type = 'broad'

            if line_type not in groups:
                groups[line_type] = []
            groups[line_type].append((idx, line))

        # Track line order for validation mapping
        line_order: List[int] = []

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
                'TieDispersion': False,  # Don't tie FWHM in validation - test individual recovery
                'Species': species_list,
            }

        # Add continuum configuration if not using default linear
        if self.continuum is not None:
            # Handle composite continuum (list of components)
            if isinstance(self.continuum, list):
                config['continuum'] = []
                for cont_component in self.continuum:
                    if cont_component.continuum_type == 'blackbody':
                        config['continuum'].append({
                            'type': 'blackbody',
                            'pivot_micron': cont_component.pivot_micron,
                            'temp_type': cont_component.temp_type,
                        })
                    elif cont_component.continuum_type == 'modified_blackbody':
                        config['continuum'].append({
                            'type': 'modified_blackbody',
                            'pivot_micron': cont_component.pivot_micron,
                            'temp_type': cont_component.temp_type,
                        })
                    elif cont_component.continuum_type == 'attenuated_blackbody':
                        config['continuum'].append({
                            'type': 'attenuated_blackbody',
                            'pivot_micron': cont_component.pivot_micron,
                            'temp_type': cont_component.temp_type,
                        })
            # Handle single continuum component
            elif self.continuum.continuum_type != 'linear':
                if self.continuum.continuum_type == 'blackbody':
                    config['continuum'] = {
                        'type': 'blackbody',
                        'pivot_micron': self.continuum.pivot_micron,
                        'temp_type': self.continuum.temp_type,
                    }
                elif self.continuum.continuum_type == 'modified_blackbody':
                    config['continuum'] = {
                        'type': 'modified_blackbody',
                        'pivot_micron': self.continuum.pivot_micron,
                        'temp_type': self.continuum.temp_type,
                    }
                elif self.continuum.continuum_type == 'attenuated_blackbody':
                    config['continuum'] = {
                        'type': 'attenuated_blackbody',
                        'pivot_micron': self.continuum.pivot_micron,
                        'temp_type': self.continuum.temp_type,
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
    ):
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
        FitResults
            Results object with file paths and samples.
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
        fit_results = NIRSpecFit(
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

        # Store fit results and samples
        self.fit_result = fit_results
        self.samples = fit_results.samples

        return fit_results

    def run(
        self,
        N: int = 500,
        num_warmup: int = 250,
        verbose: bool = True,
        sigma_tolerance: float = 3.0,
        output_dir: str | Path = 'validation_out',
    ) -> ValidationResult:
        """Convenience method to run full validation workflow.

        Executes generate_config() -> inject() -> fit() -> validate() in sequence.

        Parameters
        ----------
        N : int
            Number of MCMC samples.
        num_warmup : int
            Number of warmup samples.
        verbose : bool
            Print progress.
        sigma_tolerance : float
            Sigma tolerance for pass/fail.
        output_dir : str or Path
            Directory for output files.

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status.
        """
        # Generate config first (needed for inject to compute continuum regions)
        self.generate_config()

        # Inject lines/continuum unless continuum-only mode
        if not self.continuum_only:
            self.inject()
        else:
            # Continuum-only mode: still need to set injected_spectra for fitting
            # Just use the base spectra as-is (no injection needed)
            self.injected_spectra = self.base_spectra

        # Run fitting
        self.fit(output_dir=output_dir, N=N, num_warmup=num_warmup, verbose=verbose)

        # Validate results (only if not continuum-only)
        if not self.continuum_only:
            result = self.validate(sigma_tolerance=sigma_tolerance)
        else:
            # Continuum-only mode: create minimal validation result
            from dataclasses import dataclass

            result = ValidationResult(
                passed=True,
                metrics={},
                injected={},
                recovered={},
                uncertainties={},
                nsigma={},
                snr=np.array([]),
                line_names=[],
                sigma_tolerance=sigma_tolerance,
                details='Continuum-only mode (no line validation)',
            )
            result.fit_result = self.fit_result

        return result

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

        # Only validate lines that were actually recovered
        # (some lines may be filtered out if outside spectral range)
        n_recovered = len(flux_recovered)
        if len(ordered_lines) != n_recovered:
            logger.warning(
                f'Line count mismatch: {len(ordered_lines)} injected, {n_recovered} recovered. '
                f'Some lines may have been filtered (outside spectral range).'
            )
            # Truncate to only validate recovered lines
            ordered_lines = ordered_lines[:n_recovered]
            line_names = line_names[:n_recovered]

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
        # Store continuum model type for reporting
        continuum_model_type = None
        if self.injected_continuum is not None:
            if isinstance(self.injected_continuum, list):
                continuum_model_type = 'composite'
            else:
                continuum_model_type = self.injected_continuum.continuum_type

        # Composite continuum parameters (report but don't validate)
        if continuum_model_type == 'composite':
            # Extract parameters for both components
            # Extract composite continuum components (MBB, BB, ABB)
            composite_params = {}
            for i in range(1, 10):  # Support up to 9 components
                # Check for MBB component (modified blackbody with beta)
                mbb_amp_key = f'mbb{i}_amplitude'
                if mbb_amp_key in self.samples:
                    amp_samples = self.samples[mbb_amp_key]
                    temp_samples = self.samples[f'mbb{i}_temperature']
                    beta_samples = self.samples[f'mbb{i}_beta']

                    composite_params[f'mbb{i}_amplitude'] = float(np.median(amp_samples))
                    composite_params[f'mbb{i}_temperature'] = float(np.median(temp_samples))
                    composite_params[f'mbb{i}_beta'] = float(np.median(beta_samples))

                    composite_params[f'mbb{i}_amplitude_lo'] = float(np.percentile(amp_samples, 16))
                    composite_params[f'mbb{i}_amplitude_hi'] = float(np.percentile(amp_samples, 84))
                    composite_params[f'mbb{i}_temperature_lo'] = float(np.percentile(temp_samples, 16))
                    composite_params[f'mbb{i}_temperature_hi'] = float(np.percentile(temp_samples, 84))
                    composite_params[f'mbb{i}_beta_lo'] = float(np.percentile(beta_samples, 16))
                    composite_params[f'mbb{i}_beta_hi'] = float(np.percentile(beta_samples, 84))

                # Check for ABB component (attenuated blackbody with tau_v and alpha)
                abb_amp_key = f'abb{i}_amplitude'
                if abb_amp_key in self.samples:
                    amp_samples = self.samples[abb_amp_key]
                    temp_samples = self.samples[f'abb{i}_temperature']
                    tau_v_samples = self.samples[f'abb{i}_tau_v']
                    alpha_samples = self.samples[f'abb{i}_alpha']

                    composite_params[f'abb{i}_amplitude'] = float(np.median(amp_samples))
                    composite_params[f'abb{i}_temperature'] = float(np.median(temp_samples))
                    composite_params[f'abb{i}_tau_v'] = float(np.median(tau_v_samples))
                    composite_params[f'abb{i}_alpha'] = float(np.median(alpha_samples))

                    composite_params[f'abb{i}_amplitude_lo'] = float(np.percentile(amp_samples, 16))
                    composite_params[f'abb{i}_amplitude_hi'] = float(np.percentile(amp_samples, 84))
                    composite_params[f'abb{i}_temperature_lo'] = float(np.percentile(temp_samples, 16))
                    composite_params[f'abb{i}_temperature_hi'] = float(np.percentile(temp_samples, 84))
                    composite_params[f'abb{i}_tau_v_lo'] = float(np.percentile(tau_v_samples, 16))
                    composite_params[f'abb{i}_tau_v_hi'] = float(np.percentile(tau_v_samples, 84))
                    composite_params[f'abb{i}_alpha_lo'] = float(np.percentile(alpha_samples, 16))
                    composite_params[f'abb{i}_alpha_hi'] = float(np.percentile(alpha_samples, 84))

                # Check for plain BB component (blackbody, uncommon in composite but possible)
                bb_amp_key = f'bb{i}_amplitude'
                if bb_amp_key in self.samples:
                    amp_samples = self.samples[bb_amp_key]
                    temp_samples = self.samples[f'bb{i}_temperature']

                    composite_params[f'bb{i}_amplitude'] = float(np.median(amp_samples))
                    composite_params[f'bb{i}_temperature'] = float(np.median(temp_samples))

                    composite_params[f'bb{i}_amplitude_lo'] = float(np.percentile(amp_samples, 16))
                    composite_params[f'bb{i}_amplitude_hi'] = float(np.percentile(amp_samples, 84))
                    composite_params[f'bb{i}_temperature_lo'] = float(np.percentile(temp_samples, 16))
                    composite_params[f'bb{i}_temperature_hi'] = float(np.percentile(temp_samples, 84))

            # Don't validate composite continuum, just report
            cont_pass = True
        # Linear continuum parameters
        elif self.injected_continuum is not None and 'cont_angle' in self.samples:
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

        # Blackbody/Modified blackbody/Attenuated blackbody continuum parameters
        # Check for 'bb_', 'mbb_', and 'abb_' prefixes (model uses different prefixes)
        elif self.injected_continuum is not None and ('bb_amplitude' in self.samples or 'mbb_amplitude' in self.samples or 'abb_amplitude' in self.samples):
            # Determine which prefix to use
            if 'mbb_amplitude' in self.samples:
                amp_key = 'mbb_amplitude'
                temp_key = 'mbb_temperature'
            elif 'abb_amplitude' in self.samples:
                amp_key = 'abb_amplitude'
                temp_key = 'abb_temperature'
            else:
                amp_key = 'bb_amplitude'
                temp_key = 'bb_temperature'

            bb_amplitude_all = self.samples[amp_key]
            bb_temperature_all = self.samples[temp_key]

            bb_amplitude_recovered = float(np.median(bb_amplitude_all))
            bb_temperature_recovered = float(np.median(bb_temperature_all))

            bb_amplitude_lo = float(np.percentile(bb_amplitude_all, 16))
            bb_amplitude_hi = float(np.percentile(bb_amplitude_all, 84))
            bb_temperature_lo = float(np.percentile(bb_temperature_all, 16))
            bb_temperature_hi = float(np.percentile(bb_temperature_all, 84))

            bb_amplitude_sigma = (bb_amplitude_hi - bb_amplitude_lo) / 2
            bb_temperature_sigma = (bb_temperature_hi - bb_temperature_lo) / 2

            bb_amplitude_injected = self.injected_continuum.bb_amplitude
            bb_temperature_injected = self.injected_continuum.bb_temperature

            bb_amplitude_abs_error = abs(bb_amplitude_recovered - bb_amplitude_injected)
            bb_temperature_abs_error = abs(bb_temperature_recovered - bb_temperature_injected)

            bb_amplitude_nsigma = bb_amplitude_abs_error / max(bb_amplitude_sigma, 1e-10)
            bb_temperature_nsigma = bb_temperature_abs_error / max(bb_temperature_sigma, 1e-10)

            # Modified blackbody has beta parameter
            if 'mbb_beta' in self.samples:
                bb_beta_all = self.samples['mbb_beta']
                bb_beta_recovered = float(np.median(bb_beta_all))
                bb_beta_lo = float(np.percentile(bb_beta_all, 16))
                bb_beta_hi = float(np.percentile(bb_beta_all, 84))
                bb_beta_sigma = (bb_beta_hi - bb_beta_lo) / 2
                bb_beta_injected = self.injected_continuum.bb_beta
                bb_beta_abs_error = abs(bb_beta_recovered - bb_beta_injected)
                bb_beta_nsigma = bb_beta_abs_error / max(bb_beta_sigma, 1e-10)

            # Attenuated blackbody has tau_v and alpha parameters
            if 'abb_tau_v' in self.samples:
                bb_tau_v_all = self.samples['abb_tau_v']
                bb_tau_v_recovered = float(np.median(bb_tau_v_all))
                bb_tau_v_lo = float(np.percentile(bb_tau_v_all, 16))
                bb_tau_v_hi = float(np.percentile(bb_tau_v_all, 84))
                bb_tau_v_sigma = (bb_tau_v_hi - bb_tau_v_lo) / 2
                bb_tau_v_injected = self.injected_continuum.bb_tau_v
                bb_tau_v_abs_error = abs(bb_tau_v_recovered - bb_tau_v_injected)
                bb_tau_v_nsigma = bb_tau_v_abs_error / max(bb_tau_v_sigma, 1e-10)

            if 'abb_alpha' in self.samples:
                bb_alpha_all = self.samples['abb_alpha']
                bb_alpha_recovered = float(np.median(bb_alpha_all))
                bb_alpha_lo = float(np.percentile(bb_alpha_all, 16))
                bb_alpha_hi = float(np.percentile(bb_alpha_all, 84))
                bb_alpha_sigma = (bb_alpha_hi - bb_alpha_lo) / 2
                bb_alpha_injected = self.injected_continuum.bb_alpha if self.injected_continuum.bb_alpha is not None else -0.7
                bb_alpha_abs_error = abs(bb_alpha_recovered - bb_alpha_injected)
                bb_alpha_nsigma = bb_alpha_abs_error / max(bb_alpha_sigma, 1e-10)

            # Only check continuum for pass/fail if validate_continuum=True
            if validate_continuum:
                cont_pass = (bb_amplitude_nsigma < sigma_tolerance and
                           bb_temperature_nsigma < sigma_tolerance)
                if 'mbb_beta' in self.samples:
                    cont_pass = cont_pass and (bb_beta_nsigma < sigma_tolerance)
                if 'abb_tau_v' in self.samples:
                    cont_pass = cont_pass and (bb_tau_v_nsigma < sigma_tolerance)
                if 'abb_alpha' in self.samples:
                    cont_pass = cont_pass and (bb_alpha_nsigma < sigma_tolerance)

            # Add continuum metrics
            metrics['bb_amplitude_nsigma'] = bb_amplitude_nsigma
            metrics['bb_temperature_nsigma'] = bb_temperature_nsigma
            if 'mbb_beta' in self.samples:
                metrics['bb_beta_nsigma'] = bb_beta_nsigma
            if 'abb_tau_v' in self.samples:
                metrics['bb_tau_v_nsigma'] = bb_tau_v_nsigma
            if 'abb_alpha' in self.samples:
                metrics['bb_alpha_nsigma'] = bb_alpha_nsigma

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
            # Linear continuum was processed
            details_parts.append('\nLinear Continuum:')
            for i in range(len(cont_angle_nsigma)):
                details_parts.append(
                    f'  Region {i}: '
                    f'angle={cont_angle_nsigma[i]:.1f}σ ({"PASS" if cont_angle_nsigma[i] < sigma_tolerance else "FAIL"}), '
                    f'offset={cont_offset_nsigma[i]:.1f}σ ({"PASS" if cont_offset_nsigma[i] < sigma_tolerance else "FAIL"})'
                )

            # Add to result dicts
            injected_dict['cont_angle'] = cont_angle_injected
            injected_dict['cont_offset'] = cont_offset_injected
            injected_dict['continuum_type'] = continuum_model_type
            recovered_dict['cont_angle'] = cont_angle_recovered
            recovered_dict['cont_offset'] = cont_offset_recovered
            uncertainties_dict['cont_angle_sigma'] = cont_angle_sigma
            uncertainties_dict['cont_offset_sigma'] = cont_offset_sigma
            nsigma_dict['cont_angle'] = cont_angle_nsigma
            nsigma_dict['cont_offset'] = cont_offset_nsigma

        elif 'bb_amplitude_nsigma' in metrics:
            # Blackbody/Modified blackbody/Attenuated blackbody continuum was processed
            if 'bb_tau_v_nsigma' in metrics:
                model_name = 'Attenuated Blackbody'
            elif 'bb_beta_nsigma' in metrics:
                model_name = 'Modified Blackbody'
            else:
                model_name = 'Blackbody'
            details_parts.append(f'\n{model_name} Continuum:')
            details_parts.append(
                f'  amplitude={bb_amplitude_nsigma:.1f}σ ({"PASS" if bb_amplitude_nsigma < sigma_tolerance else "FAIL"}), '
                f'temperature={bb_temperature_nsigma:.1f}σ ({"PASS" if bb_temperature_nsigma < sigma_tolerance else "FAIL"})'
            )
            if 'bb_beta_nsigma' in metrics:
                details_parts.append(
                    f'  beta={bb_beta_nsigma:.1f}σ ({"PASS" if bb_beta_nsigma < sigma_tolerance else "FAIL"})'
                )
            if 'bb_tau_v_nsigma' in metrics:
                details_parts.append(
                    f'  tau_V={bb_tau_v_nsigma:.1f}σ ({"PASS" if bb_tau_v_nsigma < sigma_tolerance else "FAIL"})'
                )
            if 'bb_alpha_nsigma' in metrics:
                details_parts.append(
                    f'  alpha={bb_alpha_nsigma:.1f}σ ({"PASS" if bb_alpha_nsigma < sigma_tolerance else "FAIL"})'
                )

            # Add to result dicts
            injected_dict['bb_amplitude'] = bb_amplitude_injected
            injected_dict['bb_temperature'] = bb_temperature_injected
            injected_dict['continuum_type'] = continuum_model_type
            recovered_dict['bb_amplitude'] = bb_amplitude_recovered
            recovered_dict['bb_temperature'] = bb_temperature_recovered
            uncertainties_dict['bb_amplitude_lo'] = bb_amplitude_lo
            uncertainties_dict['bb_amplitude_hi'] = bb_amplitude_hi
            uncertainties_dict['bb_amplitude_sigma'] = bb_amplitude_sigma
            uncertainties_dict['bb_temperature_lo'] = bb_temperature_lo
            uncertainties_dict['bb_temperature_hi'] = bb_temperature_hi
            uncertainties_dict['bb_temperature_sigma'] = bb_temperature_sigma
            nsigma_dict['bb_amplitude'] = bb_amplitude_nsigma
            nsigma_dict['bb_temperature'] = bb_temperature_nsigma

            if 'bb_beta_nsigma' in metrics:
                injected_dict['bb_beta'] = bb_beta_injected
                recovered_dict['bb_beta'] = bb_beta_recovered
                uncertainties_dict['bb_beta_lo'] = bb_beta_lo
                uncertainties_dict['bb_beta_hi'] = bb_beta_hi
                uncertainties_dict['bb_beta_sigma'] = bb_beta_sigma
                nsigma_dict['bb_beta'] = bb_beta_nsigma

            if 'bb_tau_v_nsigma' in metrics:
                injected_dict['bb_tau_v'] = bb_tau_v_injected
                recovered_dict['bb_tau_v'] = bb_tau_v_recovered
                uncertainties_dict['bb_tau_v_lo'] = bb_tau_v_lo
                uncertainties_dict['bb_tau_v_hi'] = bb_tau_v_hi
                uncertainties_dict['bb_tau_v_sigma'] = bb_tau_v_sigma
                nsigma_dict['bb_tau_v'] = bb_tau_v_nsigma

            if 'bb_alpha_nsigma' in metrics:
                injected_dict['bb_alpha'] = bb_alpha_injected
                recovered_dict['bb_alpha'] = bb_alpha_recovered
                uncertainties_dict['bb_alpha_lo'] = bb_alpha_lo
                uncertainties_dict['bb_alpha_hi'] = bb_alpha_hi
                uncertainties_dict['bb_alpha_sigma'] = bb_alpha_sigma
                nsigma_dict['bb_alpha'] = bb_alpha_nsigma

        elif continuum_model_type == 'composite' and 'composite_params' in locals():
            # Composite continuum was processed
            details_parts.append(f'\nComposite Continuum:')
            recovered_dict.update(composite_params)
            recovered_dict['continuum_type'] = 'composite'
            injected_dict['continuum_type'] = 'composite'

            # Store injected parameters for each component
            if isinstance(self.injected_continuum, list):
                for i, cont in enumerate(self.injected_continuum, start=1):
                    # Determine component type by checking which parameters exist
                    if hasattr(cont, 'bb_tau_v') and cont.bb_tau_v is not None:
                        # Attenuated blackbody
                        prefix = f'abb{i}'
                        injected_dict[f'{prefix}_amplitude'] = float(cont.bb_amplitude)
                        injected_dict[f'{prefix}_temperature'] = float(cont.bb_temperature)
                        injected_dict[f'{prefix}_tau_v'] = float(cont.bb_tau_v)
                        # Alpha with default if not provided
                        alpha_val = cont.bb_alpha if cont.bb_alpha is not None else -0.7
                        injected_dict[f'{prefix}_alpha'] = float(alpha_val)
                    elif hasattr(cont, 'bb_beta') and cont.bb_beta is not None:
                        # Modified blackbody
                        prefix = f'mbb{i}'
                        injected_dict[f'{prefix}_amplitude'] = float(cont.bb_amplitude)
                        injected_dict[f'{prefix}_temperature'] = float(cont.bb_temperature)
                        injected_dict[f'{prefix}_beta'] = float(cont.bb_beta)
                    else:
                        # Plain blackbody
                        prefix = f'bb{i}'
                        injected_dict[f'{prefix}_amplitude'] = float(cont.bb_amplitude) if (hasattr(cont, 'bb_amplitude') and cont.bb_amplitude is not None) else 0.0
                        injected_dict[f'{prefix}_temperature'] = float(cont.bb_temperature) if (hasattr(cont, 'bb_temperature') and cont.bb_temperature is not None) else 0.0

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

    def plot_spectrum(
        self,
        ax=None,
        wavelength_range=None,
        show_fitted_regions=True,
        show_model=True,
        alpha_unfitted=0.3,
    ):
        """Plot the full spectrum with fitted vs unfitted regions.

        This is a convenience wrapper around unite.plotting.plotFullSpectrum().

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        wavelength_range : tuple, optional
            (wmin, wmax) in microns. If None, shows full spectrum.
        show_fitted_regions : bool
            If True, grays out non-fitted regions (default: True).
        show_model : bool
            If True, overplot the fitted model (default: True).
        alpha_unfitted : float
            Alpha value for non-fitted regions (default: 0.3).

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from unite.plotting import plotFullSpectrum

        if self.config is None:
            raise ValueError('Must run generate_config() before plotting')
        if self.output_dir is None:
            raise ValueError('Must run fit() before plotting')

        return plotFullSpectrum(
            config=self.config,
            rows=self.rows,
            output_dir=str(self.output_dir),
            samples=self.samples,
            spectra=self.injected_spectra,
            ax=ax,
            wavelength_range=wavelength_range,
            show_fitted_regions=show_fitted_regions,
            show_model=show_model,
            alpha_unfitted=alpha_unfitted,
        )


def inject_synthetic_lines(
    inspec: NIRSpecSpectrum,
    lines: List[SyntheticLine],
    continuum: SyntheticContinuum | List[SyntheticContinuum] | float | None = None,
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
    continuum : SyntheticContinuum or List[SyntheticContinuum] or float, optional
        Continuum model(s). If list, components are summed. If float, uses flat continuum.
        If None, uses median of existing flux.
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

    # Handle composite continuum (list of components)
    if isinstance(continuum, list):
        continuum_list = continuum
    else:
        continuum_list = [continuum]

    # Redshift factor (needed for rest-frame wavelengths)
    opz = 1.0 + spec.redshift_initial

    # Evaluate and sum continuum components
    cont_flux = np.zeros_like(wave)
    for cont_component in continuum_list:
        # Evaluate continuum based on type
        if cont_component.continuum_type == 'linear':
            # Linear continuum
            cont_centers = jnp.array(cont_component.regions.mean(axis=1))
            cont_model = optimized.linearContinua(
                jnp.array(wave),
                cont_centers,
                jnp.array(cont_component.angles),
                jnp.array(cont_component.offsets),
                jnp.array(cont_component.regions),
            ).sum(1)
            cont_flux += np.asarray(cont_model)
        elif cont_component.continuum_type == 'blackbody':
            # Blackbody continuum (rest-frame wavelengths)
            wave_rest = jnp.array(wave) / opz
            cont_model = cont_component.bb_amplitude * optimized.planck_function(
                wave_rest, cont_component.bb_temperature, cont_component.pivot_micron
            )
            cont_flux += np.asarray(cont_model)
        elif cont_component.continuum_type == 'modified_blackbody':
            # Modified blackbody continuum (rest-frame wavelengths)
            wave_rest = jnp.array(wave) / opz
            cont_model = cont_component.bb_amplitude * optimized.modified_blackbody(
                wave_rest, cont_component.bb_temperature, cont_component.bb_beta, cont_component.pivot_micron
            )
            cont_flux += np.asarray(cont_model)
        elif cont_component.continuum_type == 'attenuated_blackbody':
            # Attenuated blackbody continuum (rest-frame wavelengths)
            wave_rest = jnp.array(wave) / opz
            # Use alpha if provided, otherwise default to -0.7 (LMC-like)
            alpha = cont_component.bb_alpha if cont_component.bb_alpha is not None else -0.7
            cont_model = cont_component.bb_amplitude * optimized.attenuated_planck(
                wave_rest, cont_component.bb_temperature, cont_component.bb_tau_v, alpha, cont_component.pivot_micron
            )
            cont_flux += np.asarray(cont_model)
        else:
            raise ValueError(f'Unknown continuum type: {cont_component.continuum_type}')

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
