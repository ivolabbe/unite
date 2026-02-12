"""
Fitting functions for spectral data
"""

# Standard library
import re
import os
from copy import deepcopy

# Typing
from typing import Dict, Tuple

# Data Science
import pandas as pd

# Astropy packages
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack

# Numpyro
from numpyro import infer, optim
from numpyro.handlers import trace, seed, substitute

# JAX
import numpy as np
from jax import random, vmap, numpy as jnp

# unite
from unite.model import multiSpecModel, multiSpecModelV2
from unite.spectra import NIRSpecSpectra
from unite import utils, initial, parameters
from unite.continuum import parse_continuum_config
from unite import defaults
from unite.defaults import FittingMode

# Plotting packages
from matplotlib import pyplot


def _infer_fitting_mode(config: dict) -> FittingMode:
    """Infer fitting mode from config.

    Explicit ``config['fitting_mode']`` takes precedence.
    Otherwise: linear continuum → ``LINES``, anything else → ``FULL``.
    """
    if 'fitting_mode' in config:
        return FittingMode(config['fitting_mode'])
    cont_type = config.get('continuum', {}).get('type', 'linear').lower()
    if cont_type == 'linear':
        return FittingMode.LINES
    return FittingMode.FULL


def NIRSpecFit(
    config: dict,
    rows: Table | None = None,  # provide either rows
    spectra: NIRSpecSpectra | None = None,  # or spectra directly
    output_directory: str = 'out',
    N: int = 500,
    num_warmup: int = 250,
    backend: str = 'MCMC',
    rescale_errors=False,
    verbose=True,
    model_version: str = 'v2',  # 'v1' for original, 'v2' for continuum interface
) -> None:
    # Get the model arguments
    config, model_args = NIRSpecModelArgs(
        config, rows=rows, spectra=spectra, rescale_errors=rescale_errors, model_version=model_version
    )

    # Get rows if not provided (for saveResults)
    if rows is None:
        rows = model_args[0].rows

    # Get the random key
    rng_key = random.PRNGKey(0)

    # Fit the data
    match backend:
        case 'MCMC':
            samples, extras = MCMCFit(
                model_args, rng_key, N=N, num_warmup=num_warmup, verbose=verbose, model_version=model_version
            )
        case 'NS':
            samples, extras = NSFit(model_args, rng_key, model_version=model_version)
        case 'MAP':
            print('Warning, Experimental, Do Not Use')
            samples, extras = MAPFit(model_args, rng_key, model_version=model_version)
        case _:
            raise ValueError(f'Unknown backend: {backend}')

    # Save the results
    saveResults(config, rows, model_args, samples, extras, output_directory, model_version=model_version)

    # Plot the results
    from unite.plotting import plotResults

    plotResults(config, rows=rows, output_dir=output_directory)


def NIRSpecModelArgs(
    config: dict,
    rows: Table | None = None,
    spectra: NIRSpecSpectra | None = None,
    rescale_errors=True,
    model_version: str = 'v2',
) -> Tuple:
    """
    Get the model arguments for the NIRSpec data.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    rows : Table
        Table of the rows
    spectra : NIRSpecSpectra, optional
        Pre-loaded spectra
    rescale_errors : bool
        Whether to rescale errors per continuum region
    model_version : str
        'v1' for original model, 'v2' for continuum interface

    Returns
    -------
    tuple
        (config, model_args) where model_args depends on model_version
    """
    # Load the spectra (make a deep copy to avoid mutating the input)
    if spectra is None:
        spectra = NIRSpecSpectra(rows)
    else:
        spectra = deepcopy(spectra)

    # Apply config-specified default overrides (e.g. CONTINUUM, LINEPAD)
    defaults.apply_config_defaults(config)

    # Restrict config to what we have coverage of
    config = utils.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Generate Parameter Matrices
    matrices, linetypes_all = parameters.configToMatrices(config)

    # Determine fitting mode: explicit config value, or infer from continuum type
    fitting_mode = _infer_fitting_mode(config)

    # Compute fitting regions and initial continuum guesses
    fit_regions, cont_guesses = initial.compute_fit_regions(config, spectra, mode=fitting_mode)

    # Compute Line Centers and Equalized estimates
    line_centers, line_estimates_eq = initial.linesFluxesGuess(config, spectra, fit_regions, cont_guesses)

    # Restrict spectra to fitting regions and rescale errorbars in each region
    spectra.restrictAndRescale(config, fit_regions, rescale_errors=rescale_errors)

    # In CONTINUUM mode, mask emission-line pixels so only continuum is fitted
    # Uses defaults.CONTINUUM as the masking width (single width for all lines)
    if fitting_mode == FittingMode.CONTINUUM:
        for spectrum in spectra.spectra:
            mask = np.ones(len(spectrum.wave), dtype=bool)
            for region in fit_regions:
                mask &= spectrum.maskLines(
                    config, region,
                    broad_mask=defaults.CONTINUUM,
                    narrow_mask=defaults.CONTINUUM,
                )
            for key in ['wave', 'low', 'high', 'flux', 'err', 'valid']:
                setattr(spectrum, key, getattr(spectrum, key)[mask])
        # Remove spectra left with no pixels
        spectra.spectra = [s for s in spectra.spectra if len(s.wave) > 0]
        spectra.names = [s.name for s in spectra.spectra]

    # Skip if no data
    if len(spectra.spectra) == 0:
        raise ValueError('No Valid Data')

    # Model Args for V1 (original)
    if model_version == 'v1':
        return config, (
            spectra,
            matrices,
            linetypes_all,
            line_centers,
            line_estimates_eq,
            fit_regions,
            cont_guesses,
        )

    # Model Args for V2 (with continuum interface)
    elif model_version == 'v2':
        continuum_model = parse_continuum_config(config)
        continuum_model.initialize(spectra, cont_guesses)
        continuum_model.continuum_only = (fitting_mode == FittingMode.CONTINUUM)
        return config, (
            spectra,
            matrices,
            linetypes_all,
            line_centers,
            line_estimates_eq,
            fit_regions,
            continuum_model,
        )

    else:
        raise ValueError(f'Unknown model version: {model_version}')


def MCMCFit(
    model_args: tuple,
    rng_key: random.PRNGKey,
    N: int = 500,
    num_warmup: int = 250,
    verbose=True,
    model_version: str = 'v2',
) -> Tuple[Dict, Dict]:
    """
    Fit the NIRSpec data with MCMC.

    Parameters
    ----------
    model_args : tuple
        Model Arguements
    rng_key : random.PRNGKey
        JAX random key
    N : int, optional
        Number of samples, by default 500
    verbose : bool, optional
        Verbose, by default True
    model_version : str
        'v1' or 'v2'

    Returns
    -------
    Tuple[Dict, Dict]
        Samples and extras dictionaries
    """
    # Select model function
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # Use init_params from continuum model if available (v2 only)
    init_strategy = infer.init_to_uniform()
    if model_version == 'v2':
        continuum_model = model_args[-1]  # last element of model_args tuple
        init_params = getattr(continuum_model, 'init_params', {})
        if init_params:
            init_strategy = infer.init_to_value(values=init_params)

    # MCMC
    kernel = infer.NUTS(model_fn, init_strategy=init_strategy)
    mcmc = infer.MCMC(kernel, num_samples=N, num_warmup=num_warmup, progress_bar=verbose)
    mcmc.run(rng_key, *model_args)

    # Get the samples
    samples = mcmc.get_samples()

    # Compute relevant probabilities
    logL = computeProbs(samples, model_args, model_version=model_version)

    # Compute the WAIC (numerically stable via logsumexp)
    from scipy.special import logsumexp
    lppd = logsumexp(logL, axis=0) - np.log(logL.shape[0])
    waic = -2 * (lppd.sum() - logL.var(axis=0, ddof=1).sum())
    extras = {'WAIC': waic}

    return samples, extras


def NSFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 1000, model_version: str = 'v2'
) -> Tuple[Dict, Dict]:
    """
    Fit the NIRSpec data with Nested Sampling.

    Parameters
    ----------
    model_args : tuple
    model_version : str
        'v1' or 'v2'

    Returns
    -------
    Tuple[Dict, Dict]
        Samples and extras dictionaries
    """
    from numpyro.contrib.nested_sampling import NestedSampler

    # Select model function
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # Get number of variables
    with trace() as tr:
        with seed(model_fn, rng_seed=rng_key):
            model_fn(*model_args)
    nv = sum([v['value'].size for v in tr.values() if v['type'] == 'sample' and not v['is_observed']])

    # Nested Sampling
    constructor_kwargs = {'num_live_points': 50 * (nv + 1), 'max_samples': 50000}
    termination_kwargs = {'dlogZ': 0.01}
    NS = NestedSampler(model=model_fn, constructor_kwargs=constructor_kwargs, termination_kwargs=termination_kwargs)
    NS.run(rng_key, *model_args)

    # Get the sample
    samples = NS.get_samples(rng_key, N)

    # Compute relevant probabilities
    _ = computeProbs(samples, model_args, model_version=model_version)

    # Add log evidence to samples
    extras = {'logZ': float(NS._results.log_Z_mean), 'logZ_err': float(NS._results.log_Z_uncert)}

    return samples, extras


def MAPFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 1000, model_version: str = 'v2'
) -> Tuple[Dict, Dict]:
    """
    Fit the NIRSpec data with Maximum A Posteriori estimation.

    Parameters
    ----------
    model_args : tuple
        Model arguments
    rng_key : random.PRNGKey
        JAX random key
    N : int, optional
        Number of optimization steps
    model_version : str
        'v1' or 'v2'

    Returns
    -------
    Tuple[Dict, Dict]
        Samples and extras dictionaries
    """
    # Select model function
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # MAP Estimator
    svi = infer.SVI(
        model_fn,
        infer.autoguide.AutoDelta(model_fn),
        optim.Adam(step_size=1e-2),
        loss=infer.Trace_ELBO(),
    )

    # Run the optimization
    svi_result = svi.run(rng_key, N, *model_args)
    params, losses = svi_result.params, svi_result.losses
    params = {k.removesuffix('_auto_loc'): v for k, v in params.items()}

    # Get trace
    traced_model = trace(substitute(model_fn, data=params)).get_trace(*model_args)

    # Create compatible samples dictionary
    samples = {
        name: jnp.array(site['value'])[None, ...]  # Add sample dimension
        for name, site in traced_model.items()
        if site['type'] in ['deterministic', 'sample'] and not site.get('is_observed', False)
    }

    return samples, {'losses': losses}


def computeProbs(samples: dict, model_args: tuple, model_version: str = 'v2') -> np.ndarray:
    """Compute log likelihood and log density for samples."""
    # Select model function
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # Compute the log likelihood
    logLs = infer.util.log_likelihood(model_fn, samples, *model_args)
    for k, v in logLs.items():
        samples[k] = v
    logL = np.hstack([p for p in logLs.values()])  # Likelihood Matrix
    samples['logL'] = logL.sum(1)

    # Compute the log density
    logP = vmap(lambda s: infer.util.log_density(model_fn, model_args, {}, s)[0])(samples)
    samples['logP'] = np.array(logP)

    return logL


def saveResults(config, rows, model_args, samples, extras, output_dir, model_version: str = 'v2') -> None:
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    # Get common filename
    os.makedirs(f'{output_dir}/Results/', exist_ok=True)
    savename = f'{output_dir}/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}'

    # Unpack model args (same structure for v1 and v2, last element differs)
    spectra, _, _, _, _, fit_regions, _ = model_args

    # Correct sample units
    samples['flux_all'] = samples['flux_all'] * (spectra.fλ_unit * spectra.λ_unit).to(
        u.Unit(1e-20 * u.erg / (u.cm * u.cm * u.s))
    )
    samples['ew_all'] = samples['ew_all'] * spectra.λ_unit.to(u.AA)

    # Add spectra wavelength to samples
    for spectrum in spectra.spectra:
        samples[f'{spectrum.name}_wavelength'] = spectrum.wave

    # Create outputs
    colnames = [
        n for n in ['lsf_scale', 'PRISM_flux', 'PRISM_offset', 'logL', 'logP'] if n in samples.keys()
    ]
    out = Table([samples[name] for name in colnames], names=colnames)

    # Add fitting regions and error scales to samples
    samples['fit_regions'] = np.array(fit_regions)
    samples['cont_regs'] = samples['fit_regions']  # backward compat alias
    if hasattr(spectra.spectra[0], 'errscales'):
        samples.update(
            {f'{spectrum.name}_errscales': np.array(spectrum.errscales) for spectrum in spectra.spectra}
        )

    # Save all samples as npz
    np.savez(f'{savename}_full.npz', **samples)

    # Get names of the lines
    line_names = [
        (
            f'{group_name}_{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}'
            if group_name  # If group_name is not empty
            else f'{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}'
        )
        for group_name, group in config['Groups'].items()
        for species in group['Species']
        for line in species['Lines']
    ]

    # Make line names unique by appending index suffix for duplicates
    seen = {}
    unique_names = []
    for name in line_names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f'{name}_{seen[name]}')
        else:
            seen[name] = 0
            unique_names.append(name)
    line_names = unique_names

    # Append line parameter samples
    for colname, unit in zip(
        ['redshift', 'flux', 'fwhm', 'ew'],
        [u.dimensionless_unscaled, u.Unit(1e-20 * u.erg / u.cm**2 / u.s), u.km / u.s, u.AA],
    ):
        data = np.array(samples[f'{colname}_all'].T.tolist()) * unit
        out_part = Table(data.T, names=[f'{line}_{colname}' for line in line_names])
        out = hstack([out, out_part])

    # Append LSF samples
    for spectrum in spectra.spectra:
        data = np.array(samples[f'{spectrum.name}_lsf'].T.tolist()) * spectra.λ_unit
        out_part = Table(data.T, names=[f'{spectrum.name}_{line}_lsf' for line in line_names])
        out = hstack([out, out_part])

    # Create extra table
    extra = Table([[v] for v in extras.values()], names=extras.keys())

    # Create HDUList
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(out, name='PARAMS'), fits.BinTableHDU(extra, name='EXTRAS')]
    )

    # Save the summary
    hdul.writeto(f'{savename}_summary.fits', overwrite=True)

    # Create Summary CSV
    qs = [0.16, 0.5, 0.84]
    df = pd.concat([t.to_pandas().quantile(qs).T for t in [out, extra]], axis=0)
    df.columns = ['P16', 'P50', 'P84']
    df.to_csv(f'{savename}_summary.csv')


def get_components_fit(
    config: dict, model_args: tuple, samples: dict, method: str = 'median', model_version: str = 'v2'
) -> Tuple[Dict[str, Dict[str, jnp.ndarray]], dict]:
    """
    Reconstruct the individual lines and continuum for all components.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    model_args : tuple
        Arguments for the model
    samples : dict
        Samples from the MCMC
    method : str, optional
        Method to select parameters ('median' or 'max_prob'), by default 'median'
    model_version : str
        'v1' or 'v2'

    Returns
    -------
    components : Dict[str, Dict[str, jnp.ndarray]]
        Dictionary of components for each spectrum.
        Keys are spectrum names.
        Values are dictionaries with keys 'lines' (n_pixels, n_lines) and 'continuum' (n_pixels,).
    config : dict
        The restricted configuration dictionary with 'Index' keys added, matching the model.
    """
    # Select model function
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # Ensure config matches model
    spectra = model_args[0]
    config = utils.restrictConfig(config, spectra)
    parameters.configToMatrices(config)  # Adds 'Index' in place

    # Select parameters
    if method == 'median':
        params = {k: jnp.median(v, axis=0) for k, v in samples.items()}
    elif method == 'max_prob':
        idx = jnp.argmax(samples['logP'])
        params = {k: v[idx] for k, v in samples.items()}
    else:
        raise ValueError(f'Unknown method: {method}')

    # Filter out deterministic sites to force re-computation
    excluded_suffixes = ('_model', '_lines', '_cont', '_lsf', '_z_all')
    excluded_keys = {'flux_all', 'redshift_all', 'fwhm_all', 'ew_all', 'cont_center', 'logP'}

    params = {
        k: v for k, v in params.items() if k not in excluded_keys and not k.endswith(excluded_suffixes)
    }

    # Run model with trace
    with seed(rng_seed=0):
        with substitute(data=params):
            with trace() as tr:
                model_fn(*model_args)

    # Extract components
    spectra = model_args[0]
    components = {}
    for spec in spectra.spectra:
        lines = tr[f'{spec.name}_lines']['value']
        wave = tr[f'{spec.name}_wave']['value']
        continuum = tr[f'{spec.name}_cont']['value']
        model = tr[f'{spec.name}_model']['value']
        components[spec.name] = {'wave': wave, 'lines': lines, 'continuum': continuum, 'model': model}

    return components, config
