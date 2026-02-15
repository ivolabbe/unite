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


def fwhm_prior(linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None) -> dist.Distribution:
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


def redshift_prior(linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None) -> dist.Distribution:
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


def flux_prior(linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None) -> dist.Distribution:
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


def tilt_prior(offset_guess: jnp.ndarray) -> dist.Distribution:
    """Uniform prior for the normalized continuum tilt.

    The tilt is the flux change from region center to edge, so it has
    the same units as the flux.  ``tilt = offset`` means the flux doubles
    from center to edge — already an extreme slope.

    Parameters
    ----------
    offset_guess : jnp.ndarray
        Continuum height guess (sets the scale).

    Returns
    -------
    dist.Distribution
    """
    scale = jnp.abs(offset_guess)
    return dist.Uniform(low=-scale, high=scale)


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


def lsf_scale_prior(mean: float = 1.2, sig: float = 0.1, cutoff: float = 3.0) -> dist.Distribution:
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

    return dist.TruncatedNormal(loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig)


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

    return dist.TruncatedNormal(loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig)


def temperature_prior(low: float = 1000.0, high: float = 30000.0) -> dist.Distribution:
    """
    Return a uniform prior for blackbody temperature

    Parameters
    ----------
    low : float
        Lower bound in Kelvin
    high : float
        Upper bound in Kelvin

    Return
    ------
    dist.Distribution
        Prior distribution for temperature
    """
    return dist.Uniform(low=low, high=high)


def chebyshev_prior(offset_guess: jnp.ndarray) -> dist.Distribution:
    """Uniform prior for Chebyshev curvature coefficient (c2).

    Scale is set relative to the continuum height guess — curvature
    that doubles the flux is already extreme.

    Parameters
    ----------
    offset_guess : jnp.ndarray
        Continuum height guess (sets the scale).

    Returns
    -------
    dist.Distribution
    """
    scale = jnp.abs(offset_guess)
    return dist.Uniform(low=-scale, high=scale)


def amplitude_prior(guess: float) -> dist.Distribution:
    """
    Return a lognormal prior for continuum amplitude

    Parameters
    ----------
    guess : float
        Initial guess for amplitude (median of continuum height guesses)

    Return
    ------
    dist.Distribution
        Prior distribution for amplitude
    """
    # Use log-normal prior centered on the guess with scale=1.0
    # Ensures amplitude stays positive
    return dist.LogNormal(loc=jnp.log(jnp.maximum(guess, 0.01)), scale=1.0)


def bspline_coeff_prior(guess: jnp.ndarray) -> dist.Distribution:
    """Uniform prior for B-spline coefficients, centered on LS guess.

    Parameters
    ----------
    guess : jnp.ndarray
        Initial coefficient guess from weighted least squares.

    Returns
    -------
    dist.Distribution
    """
    scale = jnp.maximum(jnp.abs(guess), 0.01)
    return dist.Uniform(low=guess - 2 * scale, high=guess + 2 * scale)


def bernstein_coeff_prior(guess: jnp.ndarray) -> dist.Distribution:
    """LogNormal prior for Bernstein coefficients (positive).

    Parameters
    ----------
    guess : jnp.ndarray
        Initial coefficient guess from NNLS.

    Returns
    -------
    dist.Distribution
    """
    return dist.LogNormal(loc=jnp.log(jnp.maximum(guess, 0.01)), scale=1.0)
