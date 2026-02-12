"""
Continuum models for spectral fitting.

Provides continuum model classes for different continuum types:
- LinearContinuum: piecewise linear (default, existing behavior)
- BlackbodyContinuum: pure Planck blackbody
- ModifiedBlackbodyContinuum: A * B_λ(T) * (λ/λ₀)^β
- AttenuatedBlackbodyContinuum: A * B_λ(T) * exp(-τ_V * [(λ/λ_V)^α − (pivot/λ_V)^α])

Each model separates config (constructor) from data calibration (initialize).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist

from unite import priors, optimized, defaults
from unite.spectra import Spectra

log = logging.getLogger(__name__)


class ContinuumModel(ABC):
    """Abstract base class for continuum models.

    All continuum models evaluate in REST-FRAME wavelengths.
    model.py handles the redshift conversion.
    """

    init_params: Dict[str, jnp.ndarray] = {}

    @abstractmethod
    def initialize(self, spectra: Spectra, cont_guesses: jnp.ndarray) -> None:
        """Calibrate the model from data (amplitude guess, pivot, etc.).

        Sets ``self.init_params`` with smart starting values for MCMC.

        Parameters
        ----------
        spectra : Spectra
            Spectra object (for wavelengths and redshift)
        cont_guesses : jnp.ndarray
            Median flux per fitting region
        """
        pass

    @abstractmethod
    def sample_params(self, sample_fn, fit_regions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Sample continuum parameters using NumPyro sample function."""
        pass

    @abstractmethod
    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        fit_regions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate continuum at REST-FRAME wavelengths."""
        pass

    def _median_rest_wavelength(self, spectra: Spectra) -> float:
        """Compute median rest-frame wavelength from spectra."""
        wave_obs = np.concatenate([s.wave for s in spectra.spectra])
        return float(np.median(wave_obs)) / (1 + spectra.redshift_initial)

    def _extract_data(self, spectra: Spectra) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract rest-frame wavelengths, flux, and errors from spectra.

        Returns only valid (finite, positive-error) pixels.
        """
        z = spectra.redshift_initial
        wave = np.concatenate([s.wave for s in spectra.spectra]) / (1 + z)
        flux = np.concatenate([s.flux for s in spectra.spectra])
        err = np.concatenate([s.err for s in spectra.spectra])
        valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        return wave[valid], flux[valid], err[valid]

    @staticmethod
    def _optimal_amplitude(flux: np.ndarray, model_shape: np.ndarray,
                           err: np.ndarray) -> float:
        """Solve for optimal amplitude A via weighted least squares.

        Minimizes chi² = Σ((flux - A*model_shape) / err)²
        """
        w = 1.0 / err**2
        return float(np.sum(flux * model_shape * w) / np.sum(model_shape**2 * w))

    @staticmethod
    def _grid_interior(lo: float, hi: float, n: int) -> np.ndarray:
        """Linspace strictly inside (lo, hi), avoiding prior boundaries."""
        eps = (hi - lo) * 0.01
        return np.linspace(lo + eps, hi - eps, n)


class NoContinuum(ContinuumModel):
    """No continuum — assumes continuum has been pre-subtracted."""

    def initialize(self, spectra: Spectra, cont_guesses: jnp.ndarray) -> None:
        pass

    def sample_params(self, sample_fn, fit_regions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {}

    def evaluate(
        self,
        wave: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        fit_regions: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.zeros_like(wave)


class LinearContinuum(ContinuumModel):
    """Piecewise linear continuum model (existing default behavior)."""

    def initialize(self, spectra: Spectra, cont_guesses: jnp.ndarray) -> None:
        self.cont_guesses = cont_guesses
        # No init_params: linear model converges reliably with default init

    def sample_params(self, sample_fn, fit_regions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Sample linear continuum parameters (tilt and offset per region)."""
        from numpyro import plate

        Nc = len(fit_regions)
        with plate(f'Nc = {Nc}', Nc):
            tilts = sample_fn('cont_tilt', priors.tilt_prior(self.cont_guesses))
            offsets = sample_fn('cont_offset', priors.height_prior(self.cont_guesses))

        return {'cont_tilt': tilts, 'cont_offset': offsets}

    def evaluate(
        self,
        wave: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        fit_regions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate piecewise linear continuum."""
        return optimized.linearContinuaNorm(
            wave,
            params['cont_tilt'],
            params['cont_offset'],
            fit_regions,
        ).sum(1)


class BlackbodyContinuum(ContinuumModel):
    """Planck blackbody continuum model."""

    def __init__(self, temp_bounds: tuple = None):
        self.temp_bounds = temp_bounds or defaults.temperature['default']

    def initialize(self, spectra: Spectra, cont_guesses: jnp.ndarray) -> None:
        self.pivot_micron = self._median_rest_wavelength(spectra)
        self.amplitude_guess = float(cont_guesses.mean()) if len(cont_guesses) > 0 else 1.0

        # Quick grid fit over temperature
        wave, flux, err = self._extract_data(spectra)
        T_lo, T_hi = self.temp_bounds

        best_chi2, best_T, best_A = np.inf, np.mean(self.temp_bounds), self.amplitude_guess
        for T in self._grid_interior(T_lo, T_hi, 10):
            shape = np.array(optimized.planck_function(
                jnp.array(wave), jnp.array(T), self.pivot_micron))
            A = self._optimal_amplitude(flux, shape, err)
            chi2 = np.sum(((flux - A * shape) / err)**2)
            if chi2 < best_chi2:
                best_chi2, best_T, best_A = chi2, T, A

        self.amplitude_guess = best_A
        self.init_params = {
            'bb_amplitude': jnp.array(best_A),
            'bb_temperature': jnp.array(best_T),
        }
        log.info(f"BB init: T={best_T:.0f} K, A={best_A:.2f}")

    def sample_params(self, sample_fn, fit_regions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        amplitude = sample_fn('bb_amplitude', priors.amplitude_prior(self.amplitude_guess))
        temperature = sample_fn(
            'bb_temperature',
            dist.Uniform(low=self.temp_bounds[0], high=self.temp_bounds[1]),
        )
        return {'bb_amplitude': amplitude, 'bb_temperature': temperature}

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        fit_regions: jnp.ndarray,
    ) -> jnp.ndarray:
        return params['bb_amplitude'] * optimized.planck_function(
            wave_rest, params['bb_temperature'], self.pivot_micron,
        )


class ModifiedBlackbodyContinuum(ContinuumModel):
    """Modified blackbody: A * B_λ(T) * (λ/λ_pivot)^β.

    β < 0 suppresses the Rayleigh-Jeans tail (narrower than pure BB).
    β > 0 enhances it (broader). β = 0 recovers pure blackbody.
    """

    def __init__(self, temp_bounds: tuple = None, beta_bounds: tuple = (-2.0, 2.0)):
        self.temp_bounds = temp_bounds or defaults.temperature['default']
        self.beta_bounds = beta_bounds

    def initialize(self, spectra: Spectra, cont_guesses: jnp.ndarray) -> None:
        self.pivot_micron = self._median_rest_wavelength(spectra)
        self.amplitude_guess = float(cont_guesses.mean()) if len(cont_guesses) > 0 else 1.0

        wave, flux, err = self._extract_data(spectra)
        wave_j = jnp.array(wave)
        T_lo, T_hi = self.temp_bounds
        b_lo, b_hi = self.beta_bounds

        best_chi2 = np.inf
        best_T, best_beta, best_A = np.mean(self.temp_bounds), 0.0, self.amplitude_guess
        for T in self._grid_interior(T_lo, T_hi, 8):
            planck = np.array(optimized.planck_function(wave_j, jnp.array(T), self.pivot_micron))
            for beta in self._grid_interior(b_lo, b_hi, 5):
                shape = planck * (wave / self.pivot_micron) ** beta
                A = self._optimal_amplitude(flux, shape, err)
                chi2 = np.sum(((flux - A * shape) / err)**2)
                if chi2 < best_chi2:
                    best_chi2, best_T, best_beta, best_A = chi2, T, beta, A

        self.amplitude_guess = best_A
        self.init_params = {
            'bb_amplitude': jnp.array(best_A),
            'bb_temperature': jnp.array(best_T),
            'bb_beta': jnp.array(best_beta),
        }
        log.info(f"Modified BB init: T={best_T:.0f} K, A={best_A:.2f}, beta={best_beta:.2f}")

    def sample_params(self, sample_fn, fit_regions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        amplitude = sample_fn('bb_amplitude', priors.amplitude_prior(self.amplitude_guess))
        temperature = sample_fn(
            'bb_temperature',
            dist.Uniform(low=self.temp_bounds[0], high=self.temp_bounds[1]),
        )
        beta = sample_fn(
            'bb_beta',
            dist.Uniform(low=self.beta_bounds[0], high=self.beta_bounds[1]),
        )
        return {'bb_amplitude': amplitude, 'bb_temperature': temperature, 'bb_beta': beta}

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        fit_regions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate A * B_λ(T) * (λ/λ_pivot)^β."""
        planck = optimized.planck_function(wave_rest, params['bb_temperature'], self.pivot_micron)
        modifier = (wave_rest / self.pivot_micron) ** params['bb_beta']
        return params['bb_amplitude'] * planck * modifier


class AttenuatedBlackbodyContinuum(ContinuumModel):
    """Dust-attenuated blackbody: A * B_λ(T) * exp(-τ_V * [(λ/λ_V)^α − (pivot/λ_V)^α]).

    Extinction is normalized at the pivot wavelength so that the amplitude A
    represents the observed flux at the pivot (not the unextincted flux).
    α < 0 gives steeper extinction at short wavelengths (typical for dust).
    τ_V controls the overall optical depth at the reference wavelength λ_V.
    """

    def __init__(
        self,
        temp_bounds: tuple = None,
        tau_v_bounds: tuple = (0.0, 5.0),
        alpha_bounds: tuple = (-2.0, 0.0),
        lambda_v_micron: float = 0.55,
    ):
        self.temp_bounds = temp_bounds or defaults.temperature['default']
        self.tau_v_bounds = tau_v_bounds
        self.alpha_bounds = alpha_bounds
        self.lambda_v_micron = lambda_v_micron

    def initialize(self, spectra: Spectra, cont_guesses: jnp.ndarray) -> None:
        self.pivot_micron = self._median_rest_wavelength(spectra)
        self.amplitude_guess = float(cont_guesses.mean()) if len(cont_guesses) > 0 else 1.0

        wave, flux, err = self._extract_data(spectra)
        wave_j = jnp.array(wave)
        T_lo, T_hi = self.temp_bounds
        tau_lo, tau_hi = self.tau_v_bounds
        a_lo, a_hi = self.alpha_bounds

        ext_pivot_fn = lambda alpha: (self.pivot_micron / self.lambda_v_micron) ** alpha

        best_chi2 = np.inf
        best = {'T': np.mean(self.temp_bounds), 'tau': 0.0,
                'alpha': np.mean(self.alpha_bounds), 'A': self.amplitude_guess}

        for T in self._grid_interior(T_lo, T_hi, 8):
            planck = np.array(optimized.planck_function(
                wave_j, jnp.array(T), self.pivot_micron))
            for tau in self._grid_interior(tau_lo, tau_hi, 6):
                for alpha in self._grid_interior(a_lo, a_hi, 5):
                    ext_d = (wave / self.lambda_v_micron) ** alpha
                    ext_p = ext_pivot_fn(alpha)
                    extinction = np.exp(-tau * (ext_d - ext_p))
                    shape = planck * extinction
                    A = self._optimal_amplitude(flux, shape, err)
                    chi2 = np.sum(((flux - A * shape) / err)**2)
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best = {'T': T, 'tau': tau, 'alpha': alpha, 'A': A}

        self.amplitude_guess = best['A']
        self.init_params = {
            'bb_amplitude': jnp.array(best['A']),
            'bb_temperature': jnp.array(best['T']),
            'bb_tau_v': jnp.array(best['tau']),
            'bb_alpha': jnp.array(best['alpha']),
        }
        log.info(f"Attenuated BB init: T={best['T']:.0f} K, A={best['A']:.2f}, "
                 f"tau_v={best['tau']:.2f}, alpha={best['alpha']:.2f}")

    def sample_params(self, sample_fn, fit_regions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        amplitude = sample_fn('bb_amplitude', priors.amplitude_prior(self.amplitude_guess))
        temperature = sample_fn(
            'bb_temperature',
            dist.Uniform(low=self.temp_bounds[0], high=self.temp_bounds[1]),
        )
        tau_v = sample_fn(
            'bb_tau_v',
            dist.Uniform(low=self.tau_v_bounds[0], high=self.tau_v_bounds[1]),
        )
        alpha = sample_fn(
            'bb_alpha',
            dist.Uniform(low=self.alpha_bounds[0], high=self.alpha_bounds[1]),
        )
        return {
            'bb_amplitude': amplitude, 'bb_temperature': temperature,
            'bb_tau_v': tau_v, 'bb_alpha': alpha,
        }

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        fit_regions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate A * B_λ(T) * exp(-τ_V * [(λ/λ_V)^α − (pivot/λ_V)^α]).

        Extinction is normalized at the pivot so A = observed flux at pivot.
        """
        planck = optimized.planck_function(wave_rest, params['bb_temperature'], self.pivot_micron)
        ext_data = (wave_rest / self.lambda_v_micron) ** params['bb_alpha']
        ext_pivot = (self.pivot_micron / self.lambda_v_micron) ** params['bb_alpha']
        extinction = jnp.exp(-params['bb_tau_v'] * (ext_data - ext_pivot))
        return params['bb_amplitude'] * planck * extinction


def parse_continuum_config(config: dict) -> ContinuumModel:
    """Create a ContinuumModel from config (no data needed).

    Call ``model.initialize(spectra, cont_guesses)`` afterwards to
    calibrate amplitude and pivot from the actual data.

    Parameters
    ----------
    config : dict
        UNITE configuration dictionary

    Returns
    -------
    ContinuumModel
        Uninitialized continuum model instance
    """
    if 'continuum' not in config:
        return LinearContinuum()

    cont_cfg = config['continuum']
    cont_type = cont_cfg.get('type', 'linear').lower()

    if cont_type == 'linear':
        return LinearContinuum()

    elif cont_type == 'blackbody':
        return BlackbodyContinuum(
            temp_bounds=cont_cfg.get('temperature', None),
        )

    elif cont_type == 'modified_blackbody':
        return ModifiedBlackbodyContinuum(
            temp_bounds=cont_cfg.get('temperature', None),
            beta_bounds=cont_cfg.get('beta', (-2.0, 2.0)),
        )

    elif cont_type == 'attenuated_blackbody':
        return AttenuatedBlackbodyContinuum(
            temp_bounds=cont_cfg.get('temperature', None),
            tau_v_bounds=cont_cfg.get('tau_v', (0.0, 5.0)),
            alpha_bounds=cont_cfg.get('alpha', (-2.0, 0.0)),
            lambda_v_micron=cont_cfg.get('lambda_v_micron', 0.55),
        )

    elif cont_type == 'none':
        return NoContinuum()

    else:
        raise ValueError(f'Unknown continuum type: {cont_type}')
