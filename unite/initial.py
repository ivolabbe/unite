"""
Functions for initial parameter estimation
"""

from typing import Tuple

# Astronomy packages
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np
import jax.numpy as jnp

# Spectra class
from unite import defaults
from unite.spectra import Spectra, Spectrum


def linesFluxesGuess(
    config: list,
    spectra: Spectra,
    fit_regions: jnp.ndarray,
    cont_guesses: jnp.ndarray,
    inner: u.Quantity = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Guess the line fluxes for a given configuration

    Parameters
    ----------
    spectra : Spectra
        Spectra
    config : dict
        Configuration of emission lines
    fit_regions : jnp.ndarray
        Fitting regions (observed-frame)
    cont_guesses : jnp.ndarray
        Continuum height guesses
    inner : u.Quantity, optional
        Inner region to compute the flux

    Returns
    -------
    tuple(jnp.ndarray, jnp.ndarray)
        Line centers and line flux guesses

    """
    if inner is None:
        inner = defaults.LINEPAD

    # Convert to resolution
    inner = (inner / consts.c).to(u.dimensionless_unscaled).value

    # Compute the line centers and relative strengths
    centers, strengths = jnp.array(
        [
            (li['Wavelength'], li['RelStrength'] if li['RelStrength'] is not None else 1)
            for g in config['Groups'].values()
            for s in g['Species']
            for li in s['Lines']
        ]
    ).T
    centers = jnp.array(u.Quantity(centers, config['Unit']).to(spectra.λ_unit))  # Correct units

    # Compute the relevant continuum guesses
    opz = 1 + spectra.redshift_initial
    line_conts = cont_guesses[
        jnp.argmax(
            (fit_regions[:, 0][None, :] <= centers[:, None] * opz)
            & (opz * centers[:, None] <= fit_regions[:, 1][None, :]),
            axis=1,
        )
    ]

    # Get the guesses
    guesses = jnp.array(
        [
            max([lineFluxGuess(spectrum, center, line_cont, inner) for spectrum in spectra.spectra])
            for center, line_cont in zip(centers, line_conts)
        ]
    )
    guesses = jnp.abs(guesses) / strengths  # Divide by strengths to normalize

    # For all lines that are tied, guess to the max value divided
    i = 0
    for group in config['Groups'].values():
        for species in group['Species']:
            species_guesses, species_inds = [], []
            for line in species['Lines']:
                if line['RelStrength'] is not None:
                    species_guesses.append(guesses[i])
                    species_inds.append(i)
                i += 1
            if species_guesses:
                species_guess = max(species_guesses)
                for ind in species_inds:
                    guesses = guesses.at[ind].set(species_guess)

    return centers, jnp.array(guesses)


# Line Flux Guess
def lineFluxGuess(spectrum: Spectrum, center: float, line_cont: float, inner: u.Quantity) -> float:
    """
    Compute the line flux guess as the sum of the flux in the inner region minus the continuum region guess

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum
    center : float
        Line center
    line_cont : float
        Continuum height guess
    inner : u.Quantity
        Inner region to compute the flux (R)

    Returns
    -------
    float
        Line flux guess
    """

    # Redshift the line
    innerwidth = (linewav := center * (1 + spectrum.redshift_initial)) * inner

    # Compute the mask
    imask = spectrum.coverage((linewav - innerwidth), (linewav + innerwidth))

    # Check if the mask is empty
    empty = not imask.any()
    if empty:
        imask = True

    # Estimate flux as maximum deviation from zero times the width of the region
    flux = (jnp.abs(spectrum.flux[imask]).max() * (spectrum.high[imask] - spectrum.low[imask])).sum()

    # If mask is empty, negate the sign
    if empty:
        flux = -flux

    return flux


def _line_regions(
    config: dict, spectra: Spectra, pad: u.Quantity = None
) -> jnp.ndarray:
    """Compute observed-frame regions around emission lines (merged where overlapping).

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    spectra : Spectra
        Spectra
    pad : u.Quantity, optional
        Velocity half-width around each line (default: ``defaults.CONTINUUM``)

    Returns
    -------
    jnp.ndarray
        (N_regions, 2) array of [lo, hi] in observed-frame wavelength units
    """
    if pad is None:
        pad = defaults.CONTINUUM

    # Get lines from config
    lines = np.sort(
        [
            line['Wavelength']
            for group in config['Groups'].values()
            for species in group['Species']
            for line in species['Lines']
        ]
    ) * u.Unit(config['Unit'])

    # Compute pad as fractional wavelength
    pad = (pad / consts.c).to(u.dimensionless_unscaled).value

    # Generate per-line regions and merge overlapping ones
    allregs = lines[:, np.newaxis] + np.array([-1, 1]) * (pad * lines)[:, np.newaxis]
    merged = [allregs[0]]
    for region in allregs[1:]:
        if region[0] < merged[-1][1]:
            merged[-1][1] = region[1]
        else:
            merged.append(region)

    # Convert to target wavelength units and apply redshift
    regs_rest = jnp.array([r.to(spectra.λ_unit).value for r in merged])
    return regs_rest * (1 + spectra.redshift_initial)


def _full_region(spectra: Spectra) -> jnp.ndarray:
    """Single region spanning the full wavelength range of all spectra.

    Returns
    -------
    jnp.ndarray
        (1, 2) array of [lo, hi] in observed-frame wavelength units
    """
    lo = min(float(s.low.min()) for s in spectra.spectra)
    hi = max(float(s.high.max()) for s in spectra.spectra)
    return jnp.array([[lo, hi]])


def _clip_to_region(
    fit_regs: jnp.ndarray, config: dict, spectra: Spectra
) -> jnp.ndarray:
    """Clip fit regions to the config ``Region`` window.

    Parameters
    ----------
    fit_regs : jnp.ndarray
        Fitting regions (N, 2) in observed-frame.
    config : dict
        Configuration with ``Region`` key ([lo, hi] in config ``Unit``, rest-frame).
    spectra : Spectra
        Spectra (used for unit conversion and redshift).

    Returns
    -------
    jnp.ndarray
        Clipped fitting regions (M, 2), M <= N.

    Raises
    ------
    ValueError
        If all regions fall outside the window.
    """
    # Convert rest-frame config region to observed-frame
    region = u.Quantity(config['Region'], config['Unit']).to(spectra.λ_unit).value
    region = region * (1 + spectra.redshift_initial)

    # Clip each region to the window, keep only overlapping ones
    clipped = []
    for reg in fit_regs:
        lo = max(float(reg[0]), region[0])
        hi = min(float(reg[1]), region[1])
        if lo < hi:
            clipped.append([lo, hi])

    if not clipped:
        raise ValueError(
            f"All fit regions fall outside config Region {config['Region']} {config['Unit']}"
        )

    return jnp.array(clipped)


def compute_fit_regions(
    config: dict,
    spectra: Spectra,
    mode: 'defaults.FittingMode | str' = defaults.FittingMode.LINES,
    pad: u.Quantity = None,
) -> jnp.ndarray:
    """Compute fitting regions (observed-frame).

    Parameters
    ----------
    config : dict
        Configuration of emission lines.
    spectra : Spectra
        Spectra.
    mode : FittingMode or str
        Which pixels to include in the likelihood:
        - ``'lines'``: regions around emission lines (current default)
        - ``'full'``: entire spectral range
        - ``'continuum'``: entire range (line masking applied later)
    pad : u.Quantity, optional
        Velocity half-width for ``'lines'`` mode.

    Returns
    -------
    jnp.ndarray
        Fitting regions (N, 2).
    """
    mode = defaults.FittingMode(mode)

    if mode == defaults.FittingMode.LINES:
        fit_regs = _line_regions(config, spectra, pad=pad)
    elif mode in (defaults.FittingMode.FULL, defaults.FittingMode.CONTINUUM):
        fit_regs = _full_region(spectra)
    else:
        raise ValueError(f'Unknown fitting mode: {mode}')

    if 'Region' in config:
        fit_regs = _clip_to_region(fit_regs, config, spectra)

    return fit_regs


def computeContinuumRegions(
    config: dict, spectra: Spectra, pad: u.Quantity = None
) -> jnp.ndarray:
    """Compute continuum regions around emission lines (backward-compatible wrapper).

    See :func:`compute_fit_regions` for the general interface.
    """
    return compute_fit_regions(config, spectra, mode=defaults.FittingMode.LINES, pad=pad)


def continuumHeightGuesses(
    fit_regions: jnp.ndarray,
    spectra: Spectra,
    sigma: float = 0,
) -> jnp.ndarray:
    """Guess the continuum height in each fitting region.

    Uses ``spectrum.line_mask`` (must be pre-computed via
    :meth:`~unite.spectra.Spectrum.compute_line_mask`).

    Parameters
    ----------
    fit_regions : jnp.ndarray
        Fitting regions (N, 2).
    spectra : Spectra
        Spectra with ``line_mask`` already set.
    sigma : float, optional
        Upper-bound offset in units of error.

    Returns
    -------
    jnp.ndarray
        Continuum height guesses (N,).
    """
    return jnp.array(
        [
            max(
                [
                    continuumHeightGuess(region, spectrum, sigma)
                    for spectrum in spectra.spectra
                ]
            )
            for region in fit_regions
        ]
    )


def continuumHeightGuess(
    continuum_region: jnp.ndarray, spectrum: Spectrum, sigma: float
) -> jnp.ndarray:
    """Guess the continuum height for a spectrum in one region.

    Parameters
    ----------
    continuum_region : jnp.ndarray
        Boundary of the continuum region [lo, hi].
    spectrum : Spectrum
        Spectrum with ``line_mask`` attribute.
    sigma : float
        Upper-bound offset in units of error.

    Returns
    -------
    float
        Continuum height estimate.
    """
    mask = spectrum.coverage(continuum_region[0], continuum_region[1]) & spectrum.line_mask

    # If no coverage, return very large guess, but negative so it can be overwritten by other disperser.
    if mask.sum() == 0:
        return -jnp.abs(spectrum.flux + sigma * spectrum.err).max()

    return jnp.median(spectrum.flux[mask] + sigma * spectrum.err[mask])
