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
    cont_regs: jnp.ndarray,
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
    cont_regs : jnp.ndarray
        Continuum regions
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
            (cont_regs[:, 0][None, :] <= centers[:, None] * opz)
            & (opz * centers[:, None] <= cont_regs[:, 1][None, :]),
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

    # Unit conversion: flux is in [fλ × μm], model expects [fλ × Å]
    # 1 μm = 10^4 Å, so multiply by 10^4
    flux *= 1e4

    # If mask is empty, negate the sign
    if empty:
        flux = -flux

    return flux


def computeContinuumRegions(
    config: list, spectra: Spectra, pad: u.Quantity = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the continuum regions from the configuration

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    spectra : Spectra
        Spectra
    pad : u.Quantity, optional
        Region width around the lines

    Returns
    -------
    (np.ndarray, np.ndarray)
        Continuum regions and continuum height guesses
    """
    if pad is None:
        pad = defaults.CONTINUUM

    # Check for manual region specification (continuum-only mode with regions)
    has_manual_regions = (
        'continuum' in config
        and isinstance(config['continuum'], dict)
        and 'regions' in config['continuum']
    )

    if has_manual_regions:
        # Use manual regions from config
        from unite.continuum import compute_continuum_regions
        manual_regs_obs = compute_continuum_regions(config, spectra)

        # Check if continuum-only mode
        continuum_only = config.get('continuum_only', False)

        # If continuum-only mode, exclude lines from manual regions
        # If line-fitting mode, compute line-padded regions within manual bounds
        if continuum_only and 'Groups' in config and len(config['Groups']) > 0:
            lines = np.sort(
                [
                    line['Wavelength']
                    for group in config['Groups'].values()
                    for species in group['Species']
                    for line in species['Lines']
                ]
            ) * u.Unit(config.get('Unit', 'Angstrom'))

            if len(lines) > 0:
                # Compute padding around lines
                line_pad = (pad / consts.c).to(u.dimensionless_unscaled).value
                opz = 1 + spectra.redshift_initial

                # Convert lines to observed frame microns
                lines_obs = (lines.to(spectra.λ_unit).value) * opz

                # Create line exclusion regions
                line_exclusions = []
                for line_wave in lines_obs:
                    width = line_wave * line_pad
                    line_exclusions.append([line_wave - width, line_wave + width])

                # Apply line exclusions to manual regions
                final_regions = []
                for reg_low, reg_high in cont_regs_obs:
                    current_regions = [[reg_low, reg_high]]

                    # Subtract each line exclusion
                    for exc_low, exc_high in line_exclusions:
                        new_regions = []
                        for r_low, r_high in current_regions:
                            if exc_high <= r_low or exc_low >= r_high:
                                # No overlap
                                new_regions.append([r_low, r_high])
                            elif exc_low <= r_low and exc_high >= r_high:
                                # Exclude completely covers region
                                pass
                            elif exc_low > r_low and exc_high < r_high:
                                # Exclude inside region - split
                                new_regions.append([r_low, exc_low])
                                new_regions.append([exc_high, r_high])
                            elif exc_low <= r_low:
                                # Overlap left
                                new_regions.append([exc_high, r_high])
                            else:
                                # Overlap right
                                new_regions.append([r_low, exc_low])
                        current_regions = new_regions

                    final_regions.extend(current_regions)

                if final_regions:
                    cont_regs_obs = jnp.array(final_regions)
                else:
                    cont_regs_obs = manual_regs_obs

            return cont_regs_obs, continuumHeightGuesses(cont_regs_obs, config, spectra)

        # Line-fitting mode with manual regions: compute line-padded regions within manual bounds
        # Fall through to automatic region computation below, then intersect with manual_regs_obs
        # Store manual regions for later intersection
        config['_manual_bounds'] = manual_regs_obs

    # Get lines from config
    if 'Groups' not in config or len(config['Groups']) == 0:
        # No lines, no manual regions - use full spectrum range
        wave_min = min(spec.wave.min() for spec in spectra.spectra)
        wave_max = max(spec.wave.max() for spec in spectra.spectra)
        cont_regs_obs = jnp.array([[float(wave_min), float(wave_max)]])
        return cont_regs_obs, continuumHeightGuesses(cont_regs_obs, config, spectra)

    lines = np.sort(
        [
            line['Wavelength']
            for group in config['Groups'].values()
            for species in group['Species']
            for line in species['Lines']
        ]
    ) * u.Unit(config.get('Unit', 'Angstrom'))

    # Handle empty Groups (continuum-only mode)
    if len(lines) == 0:
        # No lines to mask - use full spectrum range
        wave_min = min(spec.wave.min() for spec in spectra.spectra)
        wave_max = max(spec.wave.max() for spec in spectra.spectra)
        cont_regs_obs = jnp.array([[float(wave_min), float(wave_max)]])
        return cont_regs_obs, continuumHeightGuesses(cont_regs_obs, config, spectra)

    # Compute pad in correct units
    pad = (pad / consts.c).to(u.dimensionless_unscaled).value

    # Generate continuum regions
    allregs = lines[:, np.newaxis] + np.array([-1, 1]) * (pad * lines)[:, np.newaxis]
    cont_regs = [allregs[0]]
    for region in allregs[1:]:
        if region[0] < cont_regs[-1][1]:
            cont_regs[-1][1] = region[1]
        else:
            cont_regs.append(region)

    # Convert to correct units and redshift
    # if 'Region' in config:
    #     config_region = u.Quantity(config['Region'], config['Unit']).to(spectra.λ_unit)
    #     cont_regs = [
    #         u.Quantity([np.maximum(reg[0], config_region[0]), np.minimum(reg[1], config_region[1])])
    #         for reg in cont_regs
    #     ]

    cont_regs_rest = jnp.array([cont_regs.to(spectra.λ_unit).value for cont_regs in cont_regs])
    cont_regs_obs = cont_regs_rest * (1 + spectra.redshift_initial)

    # If manual bounds were specified (line-fitting mode), intersect with them
    if '_manual_bounds' in config:
        manual_bounds = config.pop('_manual_bounds')  # Remove temporary key
        intersected_regions = []

        for auto_low, auto_high in cont_regs_obs:
            for manual_low, manual_high in manual_bounds:
                # Compute intersection
                intersect_low = max(auto_low, manual_low)
                intersect_high = min(auto_high, manual_high)

                # Only add if intersection is valid
                if intersect_low < intersect_high:
                    intersected_regions.append([intersect_low, intersect_high])

        if intersected_regions:
            cont_regs_obs = jnp.array(intersected_regions)
        # If no intersection, fall back to automatic regions (manual bounds were too restrictive)

    return cont_regs_obs, continuumHeightGuesses(cont_regs_obs, config, spectra)


def continuumHeightGuesses(
    continuum_regions: jnp.ndarray,
    config: list,
    spectra: Spectra,
    linepad: u.Quantity = None,
    sigma: float = 0,
) -> jnp.ndarray:
    """
    Guess the continuum height for different

    Parameters
    ----------
    spectra : Spectra
        Spectra
    continuum_regions : list
        List of continuum regions
    config : dict
        Configuration of emission lines
    linepad : u.Quantity, optional
        Padding to mask line
    sigma : float, optional


    Returns
    -------
    jnp.ndarray
        Array of continuum height guesses
    """
    if linepad is None:
        linepad = defaults.LINEPAD

    # Return the updated config
    return jnp.array(
        [
            max(
                [
                    continuumHeightGuess(config, continuum_regions, spectrum, linepad, sigma)
                    for spectrum in spectra.spectra
                ]
            )
            for continuum_regions in continuum_regions
        ]
    )


# Continuum Height Guess
def continuumHeightGuess(
    config: dict, continuum_region: jnp.ndarray, spectrum: Spectrum, linepad: u.Quantity, sigma: float
) -> jnp.ndarray:
    """
    Guess the continuum height for a spectrum

    Parameters
    ----------
    continuum_region : jnp.ndarray
        Boundary of the continuum region
    config : dict
        Configuration of emission lines
    linepad : u.Quantity
        Padding to mask line
    sigma : float
        Upper bound for median calculation

    Returns
    -------
    float
        Continuum Height Estimate
    """

    # Mask the lines
    mask = spectrum.maskLines(config, continuum_region, linepad)

    # If no coverage, return very large guess, but negative so it can be overwritten by other disperser.
    if mask.sum() == 0:
        return -jnp.abs(spectrum.flux + sigma * spectrum.err).max()

    # Compute the median Nsigma upper bound
    return jnp.median(spectrum.flux[mask] + sigma * spectrum.err[mask])
