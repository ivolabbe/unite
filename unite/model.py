"""
Multi-Spectrum Model
"""

# Standard Imports
from typing import Final, Tuple, List

# Astropy
from astropy import units as u, constants as consts

# JAX
from jax import numpy as jnp
from jax.experimental.sparse import BCOO

# Bayesian Inference
from numpyro import factor, plate, sample, deterministic as determ, distributions as dist

# unite
from unite.spectra import Spectra
from unite import priors, optimized, defaults
from unite.calibration import NIRSpecCalibration

# Speed of light
C: Final[float] = consts.c.to(u.km / u.s).value


# Define the model
def multiSpecModel(
    spectra: Spectra,
    matrices: Tuple[List[BCOO], List[BCOO], List[BCOO]],
    linetypes_all: Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]],
    line_centers: jnp.ndarray,
    line_estimates_eq: jnp.ndarray,
    cont_regs: jnp.ndarray,
    cont_guesses: jnp.ndarray,
    return_components: bool = False,
) -> None:
    """
    Multi-Spectrum Model

    Parameters
    ----------
    spectra : Spectra
        Spectra to fit
    matrices : Tuple[List[BCOO], List[BCOO], List[BCOO]]
        Parameter Matrices
    linetypes_all : Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]]
        Line Type Arrays
    line_centers : jnp.ndarray
        Line centers
    line_estimates_eq : jnp.ndarray
        Equalized line estimates
    cont_regs : jnp.ndarray
        Continuum regions
    cont_guesses : jnp.ndarray
        Continuum guesses

    Returns
    -------
    None
    """

    # Build Spectrum Calibration
    calib = NIRSpecCalibration(spectra.names, spectra.fixed)

    # Unpack matrices
    orig, add, orig_add = matrices

    # Unpack line types
    linetypes, lts_orig, lts_add = linetypes_all

    # Map linetypes to optimized profile indices, Default to Gaussian (0)
    type_idx = jnp.zeros_like(linetypes, dtype=jnp.int32)
    type_idx = jnp.where(linetypes == defaults.LINETYPES['lorentzian'], optimized.LORENTZIAN, type_idx)
    type_idx = jnp.where(linetypes == defaults.LINETYPES['exponential'], optimized.EXPONENTIAL, type_idx)

    # Build the original parameters
    params = {}
    all_ps = (('flux', priors.flux_prior), ('redshift', priors.redshift_prior), ('fwhm', priors.fwhm_prior))
    for i, (M_orig, lt_orig, M_add, lt_add, M_orig_add, p) in enumerate(
        zip(orig, lts_orig, add, lts_add, orig_add, all_ps)
    ):
        # Unpack prior
        label, prior = p

        # Plate over the original parameters
        N_orig = M_orig.shape[0]
        with plate(f'N_{label}_orig = {N_orig}', N_orig):
            # Create the original parameters
            p_orig = sample(f'{label}_orig', prior(lt_orig))

        # Plate over the additional parameters
        N_add = M_add.shape[0]
        if N_add:
            with plate(f'N_{label}_add = {N_add}', N_add):
                # Create the additional parameters
                p_add = sample(f'{label}_add', prior(lt_add, p_orig @ M_orig_add))

            # Broadcast the parameters and sum
            params[label] = p_orig @ M_orig + p_add @ M_add
        else:
            # Broadcast the parameters
            params[label] = p_orig @ M_orig

    # Compute line fluxes
    fluxes = determ('flux_all', params['flux'] * line_estimates_eq)

    # Add initial redshift
    redshift = determ('redshift_all', params['redshift'] + spectra.redshift_initial)
    oneplusz = 1 + redshift

    # Get centers at the wavelength
    centers = line_centers * oneplusz

    # Transform fwhms into wavelength units
    fwhms = centers * determ('fwhm_all', params['fwhm']) / C

    # Plate over the continua
    Nc = len(cont_regs)  # Number of continuum regions
    with plate(f'Nc = {Nc}', Nc):
        # Continuum centers
        cont_centers = determ('cont_center', cont_regs.mean(axis=1))

        # Continuum angles
        angles = sample('cont_angle', priors.angle_prior())

        # Continuum offsets
        offsets = sample('cont_offset', priors.height_prior(cont_guesses))

    # Compute equivalent widths
    linecont = optimized.linearContinua(centers, cont_centers, angles, offsets, cont_regs).sum(1)
    determ('ew_all', fluxes / (linecont * oneplusz))

    # Loop over spectra
    if return_components:
        components = {}

    for spectrum in spectra.spectra:
        # Get the spectrum
        low, wave, high, flux, err = (jnp.array(x) for x in spectrum())

        # Get the calibration
        lsf_scale, pixel_offset, flux_scale = calib[spectrum.name]

        # Apply pixel offset
        low = low - spectrum.offset(low, pixel_offset)
        wave = wave - spectrum.offset(wave, pixel_offset)
        high = high - spectrum.offset(high, pixel_offset)
        cont_regs_shift = cont_regs - spectrum.offset(cont_regs, pixel_offset)

        wave = determ(f'{spectrum.name}_wave', wave)

        # Compute effective redshift after shift
        # centers_shift = centers - spectrum.offset(centers, pixel_offset)
        # determ(f'{spectrum.name}_z_all', (centers_shift / line_centers) - 1)

        # Get the LSF of the lines
        fwhms_lsf = determ(f'{spectrum.name}_lsf', spectrum.lsf(centers, lsf_scale))

        # Integrate pixels (note, this is total integral, not a density)
        pixints = optimized.integrate(low, high, centers, fwhms_lsf, fwhms, type_idx).T

        # Divide by bin width to compute flux density
        fλ = pixints / (high - low)[:, jnp.newaxis]

        # Multiply by line fluxes
        lines = determ(f'{spectrum.name}_lines', fluxes * fλ)

        # Compute continuum
        continuum = determ(
            f'{spectrum.name}_cont',
            optimized.linearContinua(wave, cont_centers, angles, offsets, cont_regs_shift).sum(1),
        )

        # Compute model
        model = determ(f'{spectrum.name}_model', flux_scale * (lines.sum(1) + continuum))

        # Compute likelihood
        sample(f'{spectrum.name}', dist.Normal(model, err), obs=flux)


#        components[spectrum.name] = (wave, lines, continuum, model)
#   if return_components:
#       return wave, fλ, params, (fluxes, linecont, centers, fwhms)


def multiSpecModelV2(
    spectra: Spectra,
    matrices: Tuple[List[BCOO], List[BCOO], List[BCOO]],
    linetypes_all: Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]],
    line_centers: jnp.ndarray,
    line_estimates_eq: jnp.ndarray,
    fit_regions: jnp.ndarray,
    continuum_model,  # ContinuumModel instance
    return_components: bool = False,
) -> None:
    """
    Multi-Spectrum Model V2 with continuum model interface.

    All continuum models evaluate in REST-FRAME wavelengths.

    Parameters
    ----------
    spectra : Spectra
        Spectra to fit
    matrices : Tuple[List[BCOO], List[BCOO], List[BCOO]]
        Parameter Matrices
    linetypes_all : Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]]
        Line Type Arrays
    line_centers : jnp.ndarray
        Line centers in REST-FRAME
    line_estimates_eq : jnp.ndarray
        Equalized line estimates
    fit_regions : jnp.ndarray
        Fitting regions in OBSERVED-FRAME (microns)
    continuum_model : ContinuumModel
        Continuum model instance (LinearContinuum, BlackbodyContinuum, etc.)

    Returns
    -------
    None
    """
    # Build Spectrum Calibration
    calib = NIRSpecCalibration(spectra.names, spectra.fixed)

    # Check if continuum-only mode (no line fitting)
    continuum_only = getattr(continuum_model, 'continuum_only', False)
    Nlines = len(line_centers)

    # Convert fitting regions to REST-FRAME for continuum model
    fit_regions_rest = fit_regions / (1 + spectra.redshift_initial)

    # Sample continuum parameters
    cont_params = continuum_model.sample_params(sample, fit_regions_rest)

    # Region centers
    cont_centers = determ('cont_center', fit_regions_rest.mean(axis=1))

    if not continuum_only:
        # --- Line parameters ---
        # Unpack matrices
        orig, add, orig_add = matrices

        # Unpack line types
        linetypes, lts_orig, lts_add = linetypes_all

        # Map linetypes to optimized profile indices
        type_idx = jnp.zeros_like(linetypes, dtype=jnp.int32)
        type_idx = jnp.where(linetypes == defaults.LINETYPES['lorentzian'], optimized.LORENTZIAN, type_idx)
        type_idx = jnp.where(linetypes == defaults.LINETYPES['exponential'], optimized.EXPONENTIAL, type_idx)

        # Build the line parameters
        params = {}
        all_ps = (('flux', priors.flux_prior), ('redshift', priors.redshift_prior), ('fwhm', priors.fwhm_prior))
        for i, (M_orig, lt_orig, M_add, lt_add, M_orig_add, p) in enumerate(
            zip(orig, lts_orig, add, lts_add, orig_add, all_ps)
        ):
            label, prior = p
            N_orig = M_orig.shape[0]
            with plate(f'N_{label}_orig = {N_orig}', N_orig):
                p_orig = sample(f'{label}_orig', prior(lt_orig))

            N_add = M_add.shape[0]
            if N_add:
                with plate(f'N_{label}_add = {N_add}', N_add):
                    p_add = sample(f'{label}_add', prior(lt_add, p_orig @ M_orig_add))
                params[label] = p_orig @ M_orig + p_add @ M_add
            else:
                params[label] = p_orig @ M_orig

        # Compute line fluxes
        fluxes = determ('flux_all', params['flux'] * line_estimates_eq)

        # Flux regularization: only penalize lines sharing a wavelength (degenerate pairs)
        flux_reg = getattr(continuum_model, 'flux_reg', 0.0)
        if flux_reg > 0:
            # Lines with >1 occurrence at same center wavelength
            degen = jnp.sum(line_centers[:, None] == line_centers[None, :], axis=1) > 1
            factor('flux_reg', -0.5 * jnp.sum(jnp.where(degen, params['flux']**2, 0.0)) / flux_reg**2)

        # Add initial redshift
        redshift = determ('redshift_all', params['redshift'] + spectra.redshift_initial)
        oneplusz = 1 + redshift

        # Get centers at the wavelength
        centers = line_centers * oneplusz

        # Transform fwhms into wavelength units
        fwhms = centers * determ('fwhm_all', params['fwhm']) / C

        # Compute equivalent widths using continuum at line centers (REST-FRAME)
        linecont = continuum_model.evaluate(line_centers, cont_params, fit_regions_rest)
        safe_cont = jnp.where(jnp.abs(linecont) > 0, linecont, 1.0)
        determ('ew_all', fluxes / (safe_cont * oneplusz))

    else:
        # --- Continuum-only: zero-valued line placeholders ---
        fluxes = determ('flux_all', jnp.zeros(Nlines))
        determ('redshift_all', jnp.full(Nlines, spectra.redshift_initial))
        determ('fwhm_all', jnp.zeros(Nlines))
        determ('ew_all', jnp.zeros(Nlines))

    # Loop over spectra
    for spectrum in spectra.spectra:
        # Get the spectrum
        low, wave, high, flux, err = (jnp.array(x) for x in spectrum())

        # Get the calibration
        lsf_scale, pixel_offset, flux_scale = calib[spectrum.name]

        # Apply pixel offset
        low = low - spectrum.offset(low, pixel_offset)
        wave = wave - spectrum.offset(wave, pixel_offset)
        high = high - spectrum.offset(high, pixel_offset)

        wave = determ(f'{spectrum.name}_wave', wave)

        if not continuum_only:
            # Get the LSF of the lines
            fwhms_lsf = determ(f'{spectrum.name}_lsf', spectrum.lsf(centers, lsf_scale))

            # Integrate pixels
            pixints = optimized.integrate(low, high, centers, fwhms_lsf, fwhms, type_idx).T

            # Divide by bin width to compute flux density
            fλ = pixints / (high - low)[:, jnp.newaxis]

            # Multiply by line fluxes
            lines = determ(f'{spectrum.name}_lines', fluxes * fλ)
        else:
            Npix = len(wave)
            determ(f'{spectrum.name}_lsf', jnp.zeros(Nlines))
            lines = determ(f'{spectrum.name}_lines', jnp.zeros((Npix, Nlines)))

        # Shift fitting regions by pixel offset (matches v1 cont_regs_shift behavior)
        fit_regions_shift = fit_regions - spectrum.offset(fit_regions, pixel_offset)
        fit_regions_rest_shift = fit_regions_shift / (1 + spectra.redshift_initial)

        # Compute continuum (in REST-FRAME wavelengths)
        wave_rest = wave / (1 + spectra.redshift_initial)
        continuum = determ(
            f'{spectrum.name}_cont',
            continuum_model.evaluate(wave_rest, cont_params, fit_regions_rest_shift),
        )

        # Compute model
        model = determ(f'{spectrum.name}_model', flux_scale * (lines.sum(1) + continuum))

        # Compute likelihood
        sample(f'{spectrum.name}', dist.Normal(model, err), obs=flux)
