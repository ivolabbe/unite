"""
Continuum models for spectral fitting.

Provides an abstract base class and concrete implementations for different
continuum models (linear, blackbody, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict
import jax.numpy as jnp
import numpyro.distributions as dist

from unite import priors, optimized


class ContinuumModel(ABC):
    """Abstract base class for continuum models."""

    @abstractmethod
    def get_priors(self) -> Dict[str, dist.Distribution]:
        """
        Return dict of parameter name -> prior distribution.

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to NumPyro distributions
        """
        pass

    @abstractmethod
    def evaluate(
        self, wave: jnp.ndarray, params: Dict[str, jnp.ndarray], cont_regs: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Evaluate continuum at given wavelengths.

        Parameters
        ----------
        wave : jnp.ndarray
            Wavelength values
        params : Dict[str, jnp.ndarray]
            Dictionary of sampled parameter values
        cont_regs : jnp.ndarray
            Continuum regions (for linear models)

        Returns
        -------
        jnp.ndarray
            Continuum flux values at wavelengths
        """
        pass

    @property
    @abstractmethod
    def param_names(self) -> list:
        """
        List of parameter names this model samples.

        Returns
        -------
        list
            List of parameter name strings
        """
        pass


class LinearContinuum(ContinuumModel):
    """Piecewise linear continuum model."""

    def __init__(self, cont_guesses: jnp.ndarray):
        """
        Initialize linear continuum model.

        Parameters
        ----------
        cont_guesses : jnp.ndarray
            Initial guesses for continuum heights
        """
        self.cont_guesses = cont_guesses

    @property
    def param_names(self) -> list:
        """List of parameter names."""
        return ['cont_angle', 'cont_offset']

    def get_priors(self) -> Dict[str, dist.Distribution]:
        """
        Return priors for linear continuum parameters.

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary with 'cont_angle' and 'cont_offset' priors
        """
        return {
            'cont_angle': priors.angle_prior(),
            'cont_offset': priors.height_prior(self.cont_guesses),
        }

    def evaluate(
        self, wave: jnp.ndarray, params: Dict[str, jnp.ndarray], cont_regs: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Evaluate piecewise linear continuum.

        Parameters
        ----------
        wave : jnp.ndarray
            Wavelength values
        params : Dict[str, jnp.ndarray]
            Dictionary with 'cont_angle' and 'cont_offset' arrays
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) array of [low, high] wavelength bounds

        Returns
        -------
        jnp.ndarray
            Sum of all linear continuum segments
        """
        cont_centers = cont_regs.mean(axis=1)
        return optimized.linearContinua(
            wave, cont_centers, params['cont_angle'], params['cont_offset'], cont_regs
        ).sum(1)
