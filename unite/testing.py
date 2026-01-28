"""
Testing utilities for validation spectra.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from astropy import units as u
from jax import numpy as jnp
from copy import deepcopy
from unite import optimized
from unite.model import C
from unite.spectra import NIRSpecSpectra

_HA_REST = 6562.8 * u.AA
_HB_REST = 4861.3 * u.AA


def inject_validation_gaussians(
    inspec: NIRSpecSpectra,
    continuum_level: float | None = None,
    line_fluxes: Tuple[float, float] = (50.0, 30.0),
    fwhm_kms: Tuple[float, float] = (300.0, 700.0),
    rng: np.random.Generator | None = None,
    lsf_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Inject two Gaussian emission lines (Hα, Hβ) into the first spectrum.

    The injected model is redshifted by `spectra.redshift_initial`, added on top
    of a constant continuum level, and perturbed with noise drawn from the
    spectrum's error array. The spectrum is modified in place.

    Parameters
    ----------
    spectra : NIRSpecSpectra
        Existing spectra object containing at least one spectrum.
    continuum_level : float, optional
        Constant continuum level to add. Defaults to the median of the existing
        flux values in the first spectrum.
    line_fluxes : tuple of float, optional
        Integrated line fluxes for (Hα, Hβ) in units of
        (spec.fλ_unit * Angstrom). These are converted internally to
        (spec.fλ_unit * spec.λ_unit) to match the wavelength grid units.
    fwhm_kms : tuple of float, optional
        Line FWHM values in km/s for (Hα, Hβ).
    rng : numpy.random.Generator, optional
        Random number generator for noise. If None, a deterministic generator
        is created with seed 0.

    Returns
    -------
    dict
        Dictionary with the injected model and parameters.
    """

    if rng is None:
        rng = np.random.default_rng(0)

    spec = deepcopy(inspec)
    low, wave, high, _flux, err = spec()

    if continuum_level is None:
        continuum = float(np.nanmedian(spec.flux))
    else:
        continuum = float(continuum_level)

    opz = 1.0 + spec.redshift_initial
    centers = np.array(
        [_HA_REST.to(spec.λ_unit).value * opz, _HB_REST.to(spec.λ_unit).value * opz]
    )

    fwhm_kms_arr = np.array(fwhm_kms, dtype=float)
    fwhm = centers * fwhm_kms_arr / C

    line_fluxes_arr = np.asarray(line_fluxes, dtype=float)
    line_fluxes_arr /= spec.λ_unit.to(u.AA)
    line_fluxes_j = jnp.asarray(line_fluxes_arr)
    low_j = jnp.asarray(low)
    high_j = jnp.asarray(high)
    centers_j = jnp.asarray(centers)
    fwhm_j = jnp.asarray(fwhm)
    type_idx = jnp.zeros(centers_j.shape[0], dtype=jnp.int32)

    lsf = spec.lsf(centers, lsf_scale)
    lsf_j = jnp.asarray(lsf)

    for c, l, f, ls in zip(centers_j, line_fluxes_j, fwhm_j, lsf_j):
        lsf_R = c / ls
        print(
            f'Line at {c:.2f} with flux {l:.2f}, FWHM {f:.3f}, LSF {ls:.4f} ({lsf_R:.1f})'
        )
    pixints = optimized.integrate(low_j, high_j, centers_j, lsf_j, fwhm_j, type_idx).T
    f_lambda = pixints / (high_j - low_j)[:, jnp.newaxis]
    line_model = f_lambda * line_fluxes_j
    model_flux = np.asarray(continuum + line_model.sum(axis=1))

    noise = rng.normal(0.0, err)
    spec.flux = model_flux + noise

    return spec
