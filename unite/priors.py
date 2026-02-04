"""
Module for defining the priors for the model parameters
"""

# Typing
from typing import Optional

# JAX packages
from jax import numpy as jnp

# Bayesian Inference
from numpyro import distributions as dist

# unite
from unite import defaults


def fwhm_prior(
    linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None
) -> dist.Distribution:
    """
    Return a fwhm prior based on the linetype

    Parameters
    ----------
    linetypes : jnp.ndarray
        Integer array of line types
    orig : jnp.ndarray, optional
        Original values for additional lines

    Return
    ------
    dist.Distribution
        Prior distribution
    """

    # Get the low and high bounds
    fwhm = defaults.convertToArray(defaults.fwhm)
    low, high = fwhm[linetypes].T

    # If there are original values, set the low and high bounds
    if orig is not None:
        # Broad line must be 100 km/s higher than the original
        is_broad = (
            (linetypes == defaults.LINETYPES['broad'])
            | (linetypes == defaults.LINETYPES['lorentzian'])
            | (linetypes == defaults.LINETYPES['exponential'])
        )

        low = jnp.where(is_broad, orig + 100, low)

    return dist.Uniform(low=low, high=high)


def redshift_prior(
    linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None
) -> dist.Distribution:
    """
    Return a redshift prior based on the linetype

    Parameters
    ----------
    linetypes : jnp.ndarray
        Integer array of line types
    orig : jnp.ndarray, optional
        Original values for additional lines

    Return
    ------
    dist.Distribution
        Prior distribution
    """

    # Get the low and high bounds
    redshift = defaults.convertToArray(defaults.redshift)
    low, high = redshift[linetypes].T

    # If there are original values, set the low and high bounds
    if orig is not None:
        # Outflow line must be blueshifted relative to the original
        high = jnp.where(linetypes == defaults.LINETYPES['outflow'], orig, high)

    return dist.Uniform(low=low, high=high)


def flux_prior(
    linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None
) -> dist.Distribution:
    """
    Return a flux prior based on the linetype

    Parameters
    ----------
    linetypes : jnp.ndarray
        Integer array of line types
    orig : jnp.ndarray, optional
        Original values for additional lines

    Return
    ------
    dist.Distribution
        Prior distribution
    """

    # Get the low and high bounds
    flux = defaults.convertToArray(defaults.flux)
    low, high = flux[linetypes].T

    return dist.Uniform(low=low, high=high)


def angle_prior() -> dist.Distribution:
    """
    Return a uniform prior for the angle of the angle of the continuum

    Parameters
    ----------
    None

    Return
    ------
    dist.Distribution
        Prior distribution for the angle of the continuum
    """

    return dist.Uniform(low=-jnp.pi / 2, high=jnp.pi / 2)


def height_prior(height_guess: float) -> dist.Distribution:
    """
    Return a uniform prior for the height of the continuum

    Parameters
    ----------
    intercept_guess : float
        Initial guess for the height of the continuum

    Return
    ------
    dist.Distribution
        Prior distribution for the height of the continuum
    """

    low = jnp.where(height_guess < 0, 2 * height_guess, -height_guess)
    high = jnp.where(height_guess < 0, -2 * height_guess, 2 * height_guess)
    return dist.Uniform(low=low, high=high)


def lsf_scale_prior(
    mean: float = 1.2, sig: float = 0.1, cutoff: float = 3.0
) -> dist.Distribution:
    """
    Return a truncated normal prior for the lsf scale
    Centered on 1.2 with a standard deviation of 0.1, but truncated at 3σ

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    sig : float, optional
        Standard deviation of the prior
    cutoff : float, optional
        Sigma cutoff for the prior


    Return
    ------
    dist.Distribution
        Prior distribution for the lsf scale
    """

    return dist.TruncatedNormal(
        loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig
    )


def pixel_offset_prior(mean: float = 0.2, half_width: float = 0.5) -> dist.Distribution:
    """
    Return a uniform prior for the pixel offset

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    half_width : float, optional
        Half width of the prior

    Return
    ------
    dist.Distribution
        Prior distribution for the pixel offset
    """

    return dist.Uniform(low=mean - half_width, high=mean + half_width)


def flux_scale_prior(mean=1.1, sig=0.2, cutoff=3.0) -> dist.Distribution:
    """
    Return a truncated normal prior for the flux scale

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    sig : float, optional
        Standard deviation of the prior
    cutoff : float, optional
        Sigma cutoff for the prior

    Return
    ------
    dist.Distribution
        Prior distribution for the flux scale
    """

    return dist.TruncatedNormal(
        loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig
    )


def temperature_prior(temp_type: str = 'default') -> dist.Distribution:
    """
    Return a uniform prior for blackbody temperature

    Parameters
    ----------
    temp_type : str, optional
        Temperature range type ('hot', 'warm', 'default')

    Return
    ------
    dist.Distribution
        Prior distribution for temperature in Kelvin
    """
    low, high = defaults.temperature[temp_type]
    return dist.Uniform(low=low, high=high)


def beta_prior() -> dist.Distribution:
    """
    Return a uniform prior for emissivity index beta

    Parameters
    ----------
    None

    Return
    ------
    dist.Distribution
        Prior distribution for beta (emissivity index)
    """
    low, high = defaults.beta['default']
    return dist.Uniform(low=low, high=high)


def amplitude_prior(guess: float) -> dist.Distribution:
    """
    Return a lognormal prior for continuum amplitude

    Parameters
    ----------
    guess : float
        Initial guess for amplitude

    Return
    ------
    dist.Distribution
        Prior distribution for amplitude
    """
    return dist.LogNormal(loc=jnp.log(jnp.maximum(guess, 0.01)), scale=1.0)


def tau_v_prior(tau_type: str = 'default') -> dist.Distribution:
    """
    Prior for V-band optical depth τ_V.

    Parameters
    ----------
    tau_type : str
        Prior type: 'low', 'moderate', 'high', or 'default'

    Returns
    -------
    dist.Distribution
        Prior distribution for optical depth
    """
    low, high = defaults.tau_v[tau_type]
    return dist.Uniform(low=low, high=high)


def alpha_atten_prior(alpha_type: str = 'default') -> dist.Distribution:
    """
    Prior for attenuation power-law slope α.

    Parameters
    ----------
    alpha_type : str
        Prior type: 'mw' (Milky Way), 'lmc', 'smc', or 'default' (full range)

    Returns
    -------
    dist.Distribution
        Prior distribution for attenuation slope
    """
    low, high = defaults.alpha_atten[alpha_type]
    return dist.Uniform(low=low, high=high)
