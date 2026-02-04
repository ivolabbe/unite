"""
Continuum models for spectral fitting.

Provides continuum model classes for different continuum types:
- LinearContinuum: piecewise linear (default, existing behavior)
- BlackbodyContinuum: pure Planck blackbody
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import jax.numpy as jnp
import numpyro.distributions as dist

from unite import priors, optimized, defaults


class ContinuumModel(ABC):
    """Abstract base class for continuum models.

    All continuum models evaluate in REST-FRAME wavelengths.
    model.py handles the redshift conversion.
    """

    @abstractmethod
    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Sample continuum parameters using NumPyro sample function.

        Parameters
        ----------
        sample_fn : callable
            NumPyro sample function (numpyro.sample)
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) in REST-FRAME wavelengths

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary of sampled parameter values
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate continuum at REST-FRAME wavelengths.

        Parameters
        ----------
        wave_rest : jnp.ndarray
            REST-FRAME wavelength values in microns
        params : Dict[str, jnp.ndarray]
            Dictionary of sampled parameter values
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) in REST-FRAME wavelengths

        Returns
        -------
        jnp.ndarray
            Continuum flux values at wavelengths
        """
        pass


class LinearContinuum(ContinuumModel):
    """Piecewise linear continuum model (existing default behavior)."""

    def __init__(self, cont_guesses: jnp.ndarray):
        """Initialize linear continuum model.

        Parameters
        ----------
        cont_guesses : jnp.ndarray
            Initial guesses for continuum heights per region
        """
        self.cont_guesses = cont_guesses

    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Sample linear continuum parameters (angle and offset per region)."""
        from numpyro import plate

        Nc = len(cont_regs)
        with plate(f'Nc = {Nc}', Nc):
            angles = sample_fn('cont_angle', priors.angle_prior())
            offsets = sample_fn('cont_offset', priors.height_prior(self.cont_guesses))

        return {'cont_angle': angles, 'cont_offset': offsets}

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate piecewise linear continuum."""
        cont_centers = cont_regs.mean(axis=1)
        return optimized.linearContinua(
            wave_rest,
            cont_centers,
            params['cont_angle'],
            params['cont_offset'],
            cont_regs,
        ).sum(1)


class BlackbodyContinuum(ContinuumModel):
    """Planck blackbody continuum model."""

    def __init__(
        self,
        amplitude_guess: float,
        temp_bounds: tuple = None,
        pivot_micron: float = 0.5,
    ):
        """Initialize blackbody continuum model.

        Parameters
        ----------
        amplitude_guess : float
            Initial guess for amplitude (median of continuum heights)
        temp_bounds : tuple, optional
            Temperature bounds (low, high) in Kelvin. Default: (1000, 30000)
        pivot_micron : float
            Pivot wavelength for normalization (default 0.5 microns)
        """
        self.amplitude_guess = amplitude_guess
        self.temp_bounds = temp_bounds or defaults.temperature['default']
        self.pivot_micron = pivot_micron

    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Sample blackbody parameters (amplitude and temperature)."""
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
        cont_regs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate blackbody at REST-FRAME wavelengths."""
        return params['bb_amplitude'] * optimized.planck_function(
            wave_rest,
            params['bb_temperature'],
            self.pivot_micron,
        )


def parse_continuum_config(config: dict, cont_guesses: jnp.ndarray) -> ContinuumModel:
    """Parse continuum config and return appropriate model.

    Config format examples:
        # Default linear (no continuum key, or explicit)
        config = {...}  # no 'continuum' key
        config = {'continuum': {'type': 'linear'}}

        # Blackbody
        config = {
            'continuum': {
                'type': 'blackbody',
                'temperature': (3000, 10000),  # optional bounds
                'pivot_micron': 0.5,  # optional
            }
        }

    Parameters
    ----------
    config : dict
        UNITE configuration dictionary
    cont_guesses : jnp.ndarray
        Initial guesses for continuum heights

    Returns
    -------
    ContinuumModel
        Appropriate continuum model instance
    """
    if 'continuum' not in config:
        return LinearContinuum(cont_guesses)

    cont_cfg = config['continuum']
    cont_type = cont_cfg.get('type', 'linear').lower()

    if cont_type == 'linear':
        return LinearContinuum(cont_guesses)

    elif cont_type == 'blackbody':
        amp_guess = float(cont_guesses.mean()) if len(cont_guesses) > 0 else 1.0
        temp_bounds = cont_cfg.get('temperature', None)
        pivot = cont_cfg.get('pivot_micron', 0.5)
        return BlackbodyContinuum(amp_guess, temp_bounds=temp_bounds, pivot_micron=pivot)

    else:
        raise ValueError(f'Unknown continuum type: {cont_type}')
