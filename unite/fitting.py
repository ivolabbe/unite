"""
Fitting functions for spectral data
"""

# Standard library
import re
import os
import copy
from dataclasses import dataclass
from pathlib import Path

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
from unite.continuum import LinearContinuum, parse_continuum_config

# Plotting packages
from matplotlib import pyplot


@dataclass
class FitResults:
    """Results from NIRSpecFit with easy access to output files.

    Attributes
    ----------
    config : dict
        Configuration dictionary used for fitting
    rows : Table
        Input spectrum table
    output_dir : str
        Output directory path
    npz_path : str
        Path to full results NPZ file (_full.npz)
    csv_path : str
        Path to summary CSV file (_summary.csv)
    fits_path : str
        Path to summary FITS file (_summary.fits)
    samples : dict
        MCMC samples dictionary (loaded from NPZ on access)
    """

    config: dict
    rows: Table
    output_dir: str
    npz_path: str
    csv_path: str
    fits_path: str
    _samples: dict = None

    @property
    def samples(self) -> dict:
        """Lazy-load samples from NPZ file."""
        if self._samples is None:
            self._samples = dict(np.load(self.npz_path, allow_pickle=True))
        return self._samples

    def print(self) -> str:
        """Print fitted parameters in formatted table.

        Returns
        -------
        str
            Formatted ASCII table of fitted parameters with uncertainties.
            Shows individual lines with full parameter names and continuum parameters.
        """
        return display_results(self.npz_path, self.config)

    def corner(
        self, max_lines: int = 20, continuum: bool = False, figsize: tuple = (12, 12), smooth: float = 1.0
    ):
        """Create corner plot of fitted parameters.

        Parameters
        ----------
        max_lines : int
            Maximum number of lines to include (default 20).
            If more lines exist, only the brightest ones are shown.
        continuum : bool
            Include continuum parameters (default False).
        figsize : tuple
            Figure size (width, height) in inches.
        smooth : float
            Smoothing scale for 2D histograms (default 1.0).

        Returns
        -------
        fig : matplotlib.figure.Figure
            Corner plot figure.
        """
        return corner_plot(
            self.samples,
            self.config,
            max_lines=max_lines,
            continuum=continuum,
            figsize=figsize,
            smooth=smooth,
        )

    def __repr__(self) -> str:
        """String representation showing file paths."""
        return (
            f'FitResults(\n'
            f'  config_name={self.config.get("Name", "")!r}\n'
            f'  root={self.rows[0]["root"]}\n'
            f'  srcid={self.rows[0]["srcid"]}\n'
            f'  npz_path={self.npz_path!r}\n'
            f'  csv_path={self.csv_path!r}\n'
            f'  fits_path={self.fits_path!r}\n'
            f')'
        )


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
    model_version: str = 'v2',
) -> FitResults:
    # Get the model arguments
    config, model_args = NIRSpecModelArgs(
        config, rows=rows, spectra=spectra, rescale_errors=rescale_errors, model_version=model_version
    )

    # Get rows if not provided (extract from spectra)
    if rows is None:
        rows = model_args[0].rows  # spectra.rows

    # Get the random key
    rng_key = random.PRNGKey(0)

    # Fit the data
    match backend:
        case 'MCMC':
            samples, extras = MCMCFit(
                model_args,
                rng_key,
                N=N,
                num_warmup=num_warmup,
                verbose=verbose,
                model_version=model_version,
            )
        case 'NS':
            samples, extras = NSFit(model_args, rng_key, model_version=model_version)
        case 'MAP':
            print('Warning, Experimental, Do Not Use')
            samples, extras = MAPFit(model_args, rng_key, model_version=model_version)
        case _:
            raise ValueError(f'Unknown backend: {backend}')

    # Save the results
    file_paths = saveResults(config, rows, model_args, samples, extras, output_directory)

    # Plot the results (use the in-memory samples/model_args to avoid reloading)
    # Skip plotting for continuum-only mode (no lines to plot)
    if not config.get('continuum_only', False):
        from unite.plotting import plotResults
        plotResults(config, rows, output_directory, samples, model_args, model_version=model_version)

    # Return FitResults object
    return FitResults(
        config=config,
        rows=rows,
        output_dir=output_directory,
        npz_path=file_paths['npz'],
        csv_path=file_paths['csv'],
        fits_path=file_paths['fits'],
        _samples=samples,
    )


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
    model_version : str
        Model version to use ('v1' or 'v2'), default 'v2'

    Returns
    -------
    tuple
        Model arguments
    """

    # Load the spectra
    if spectra is None:
        spectra = NIRSpecSpectra(rows)
    else:
        spectra = copy.deepcopy(spectra)  # avoid modifying input spectra

    # Check for continuum-only mode
    continuum_only = config.get('continuum_only', False)

    if continuum_only:
        # Continuum-only mode: fit continuum without emission lines
        # Ensure Groups exists (empty for continuum-only)
        if 'Groups' not in config or len(config['Groups']) == 0:
            config['Groups'] = {}

        # Use computeContinuumRegions - handles both manual regions and empty Groups
        cont_regs, cont_guesses = initial.computeContinuumRegions(config, spectra)

        # Restrict spectra to continuum regions
        spectra.restrictAndRescale(config, cont_regs, rescale_errors=rescale_errors)

        # Skip if no data
        if len(spectra.spectra) == 0:
            raise ValueError('No Valid Data')

        # Parse continuum models
        continuum_models = parse_continuum_config(config, cont_guesses, spectra, cont_regs)

        # Return minimal model args for continuum-only fit
        # V2 model expects these arguments but will delegate to multiSpecModelContinuumOnly
        import jax.numpy as jnp
        from jax.experimental.sparse import BCOO

        # Create empty matrices (no lines to fit)
        # Use int32 for indices (required by BCOO)
        empty_matrix = BCOO((jnp.array([]), jnp.array([], dtype=jnp.int32).reshape(0, 2)), shape=(0, 0))
        matrices = ([empty_matrix], [empty_matrix], [empty_matrix])
        linetypes_all = (jnp.array([], dtype=jnp.int32), [jnp.array([], dtype=jnp.int32)], [jnp.array([], dtype=jnp.int32)])
        line_centers = jnp.array([])
        line_estimates_eq = jnp.array([])

        return config, (
            spectra,
            matrices,
            linetypes_all,
            line_centers,
            line_estimates_eq,
            cont_regs,
            continuum_models,
            False,  # return_components
            True,  # continuum_only flag
        )

    # Standard mode: fit emission lines + continuum
    # Restrict config to what we have coverage of
    config = utils.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Generate Parameter Matrices
    matrices, linetypes_all = parameters.configToMatrices(config)

    # Compute Continuum Regions and Initial Guesses
    cont_regs, cont_guesses = initial.computeContinuumRegions(config, spectra)

    # Compute Line Centers and Equalized estimates
    line_centers, line_estimates_eq = initial.linesFluxesGuess(config, spectra, cont_regs, cont_guesses)

    # Restrict spectra to continuum regions and rescale errorbars in each region
    spectra.restrictAndRescale(config, cont_regs, rescale_errors=rescale_errors)

    # Skip if no data
    if len(spectra.spectra) == 0:
        raise ValueError('No Valid Data')

    # Model Args for V1
    if model_version == 'v1':
        return config, (
            spectra,
            matrices,
            linetypes_all,
            line_centers,
            line_estimates_eq,
            cont_regs,
            cont_guesses,
        )
    # Model Args for V2
    elif model_version == 'v2':
        continuum_models = parse_continuum_config(config, cont_guesses, spectra, cont_regs)
        return config, (
            spectra,
            matrices,
            linetypes_all,
            line_centers,
            line_estimates_eq,
            cont_regs,
            continuum_models,
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
        Model version to use ('v1' or 'v2'), default 'v2'

    Returns
    -------
    infer.MCMC
        MCMC object
    """

    # Select model
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # MCMC kernel
    # Note: Blackbody continuum uses amplitude_guess (via LogNormal prior) for better initialization
    # Temperature initialization is via median of prior bounds - tighten bounds for better convergence
    kernel = infer.NUTS(model_fn)
    mcmc = infer.MCMC(kernel, num_samples=N, num_warmup=num_warmup, progress_bar=verbose)
    mcmc.run(rng_key, *model_args)

    # Get the samples
    samples = mcmc.get_samples()

    # Compute relevant probabilities
    logL = computeProbs(samples, model_args, model_version=model_version)

    # Compute the WAIC
    waic = -2 * (np.log(np.exp(logL).mean(axis=0)).sum() - logL.var(axis=0, ddof=1).sum())
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
        Model version to use ('v1' or 'v2'), default 'v2'

    Returns
    -------
    NestedSampler
    """

    from numpyro.contrib.nested_sampling import NestedSampler

    # Select model
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # Get number of variables
    with trace() as tr:
        with seed(model_fn, rng_seed=rng_key):
            model_fn(*model_args)
    nv = sum([v['value'].size for v in tr.values() if v['type'] == 'sample' and not v['is_observed']])

    # Nested Sampling
    constructor_kwargs = {'num_live_points': 50 * (nv + 1), 'max_samples': 50000}
    termination_kwargs = {'dlogZ': 0.01}
    NS = NestedSampler(
        model=model_fn, constructor_kwargs=constructor_kwargs, termination_kwargs=termination_kwargs
    )
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
    num_steps : int, optional
        Number of optimization steps
    model_version : str
        Model version to use ('v1' or 'v2'), default 'v2'

    Returns
    -------
    Tuple[Dict, Dict]
        Samples and extras dictionaries
    """

    # Select model
    model_fn = multiSpecModel if model_version == 'v1' else multiSpecModelV2

    # MAP Estimator
    svi = infer.SVI(
        model_fn, infer.autoguide.AutoDelta(model_fn), optim.Adam(step_size=1e-2), loss=infer.Trace_ELBO()
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
    # Select model
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


def saveResults(config, rows, model_args, samples, extras, output_dir) -> dict:
    """Save fitting results to NPZ, FITS, and CSV files.

    Returns
    -------
    dict
        Dictionary with keys 'npz', 'fits', 'csv' containing file paths.
    """
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    # Get common filename
    os.makedirs(f'{output_dir}/Results/', exist_ok=True)
    savename = f'{output_dir}/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}'

    # Unpack model args (handle both standard and continuum-only modes)
    if len(model_args) == 9:
        # Continuum-only mode (has extra continuum_only flag)
        spectra, _, _, _, _, cont_regs, _, _, _ = model_args
    else:
        # Standard mode
        spectra, _, _, _, _, cont_regs, _ = model_args

    # Correct sample units (skip if no lines)
    if 'flux_all' in samples and samples['flux_all'].size > 0:
        samples['flux_all'] = samples['flux_all'] * (spectra.fλ_unit * spectra.λ_unit).to(
            u.Unit(1e-20 * u.erg / (u.cm * u.cm * u.s))
        )
    if 'ew_all' in samples and samples['ew_all'].size > 0:
        samples['ew_all'] = samples['ew_all'] * spectra.λ_unit.to(u.AA)

    # Add spectra wavelength to samples
    for spectrum in spectra.spectra:
        samples[f'{spectrum.name}_wavelength'] = spectrum.wave

    # Create outputs
    colnames = [
        n for n in ['lsf_scale', 'PRISM_flux', 'PRISM_offset', 'logL', 'logP'] if n in samples.keys()
    ]
    out = Table([samples[name] for name in colnames], names=colnames)

    # Add continuum regions and error scales to samples
    samples['cont_regs'] = np.array(cont_regs)
    if hasattr(spectrum, 'errscales'):
        samples.update(
            {f'{spectrum.name}_errscales': np.array(spectrum.errscales) for spectrum in spectra.spectra}
        )

    # Save all samples as npz
    np.savez(f'{savename}_full.npz', **samples)

    # Get names of the lines
    # TODO: Better sanitization of line names?
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

    # Append line parameter samples (only if there are lines)
    if len(line_names) > 0 and 'redshift_all' in samples:
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

    # Append continuum samples (linear continuum)
    if 'cont_angle' in samples:
        cont_angle = samples['cont_angle']
        cont_offset = samples['cont_offset']
        # Ensure arrays are at least 1D
        cont_angle = np.atleast_1d(cont_angle)
        cont_offset = np.atleast_1d(cont_offset)

        n_regions = cont_angle.shape[1] if cont_angle.ndim > 1 else 1

        cont_data = []
        cont_names = []
        for i in range(n_regions):
            angle_data = cont_angle[:, i] if n_regions > 1 else cont_angle
            offset_data = cont_offset[:, i] if n_regions > 1 else cont_offset
            # Ensure 1D arrays
            angle_data = np.atleast_1d(np.squeeze(angle_data))
            offset_data = np.atleast_1d(np.squeeze(offset_data))
            cont_data.extend([angle_data, offset_data])
            cont_names.extend([f'cont_region{i}_angle', f'cont_region{i}_offset'])

        cont_table = Table(cont_data, names=cont_names)
        out = hstack([out, cont_table])

    # Append all blackbody continuum parameters (bb, mbb, abb with potential suffixes)
    # Pattern: {prefix}_amplitude, {prefix}_temperature, etc.
    # Prefixes: bb, bb1, bb2, ..., mbb, mbb1, mbb2, ..., abb, abb1, abb2, ...
    continuum_params = {}
    for key in samples.keys():
        # Match patterns like bb_amplitude, mbb1_temperature, abb_tau_v, etc.
        if any(key.startswith(prefix) for prefix in ['bb_', 'mbb_', 'abb_', 'bb1_', 'bb2_', 'mbb1_', 'mbb2_', 'abb1_', 'abb2_']):
            if key.endswith(('_amplitude', '_temperature', '_beta', '_tau_v', '_alpha')):
                continuum_params[key] = samples[key]

    if continuum_params:
        cont_table = Table(list(continuum_params.values()), names=list(continuum_params.keys()))
        out = hstack([out, cont_table])

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

    # Return file paths
    return {
        'npz': f'{savename}_full.npz',
        'fits': f'{savename}_summary.fits',
        'csv': f'{savename}_summary.csv',
    }


def _get_line_names(config: dict) -> list:
    """Generate line names from config (same as in saveResults)."""
    line_names = [
        (
            f'{group_name}_{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}'
            if group_name
            else f'{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}'
        )
        for group_name, group in config['Groups'].items()
        for species in group['Species']
        for line in species['Lines']
    ]
    return line_names


def display_results(npz_path: str, config: dict | None = None) -> str:
    """Display fitted parameters in a formatted table.

    Shows individual lines with full parameter names (expanded from tied groups).

    Parameters
    ----------
    npz_path : str
        Path to _full.npz file from saveResults.
    config : dict, optional
        Config dict to get line names. If None, uses generic names.

    Returns
    -------
    str
        Formatted ASCII table of fitted parameters.
    """
    from pathlib import Path

    samples = dict(np.load(npz_path, allow_pickle=True))

    # Compute quantiles
    def q(arr):
        return np.percentile(arr, [16, 50, 84], axis=0)

    lines = []
    lines.append('=' * 90)
    lines.append('  FITTED PARAMETERS (Individual Lines)')
    lines.append('=' * 90)

    # Line parameters - expand to individual lines with full names
    if 'flux_all' in samples and config is not None:
        flux_all = samples['flux_all']
        fwhm_all = samples['fwhm_all']
        z_all = samples['redshift_all']

        # Get line names from config
        line_names = _get_line_names(config)

        # Compute quantiles for each line
        flux_q = q(flux_all)
        fwhm_q = q(fwhm_all)
        z_q = q(z_all)

        # Use the actual number of lines from samples (might differ from config if restricted)
        n_lines = min(len(line_names), flux_q.shape[1])

        lines.append(f'\n{"Line Name":<50} {"Param":<10} {"P50":>12} {"[P16, P84]":>26}')
        lines.append('-' * 90)

        for i in range(n_lines):
            name = line_names[i]
            # Truncate name if too long
            display_name = name if len(name) <= 48 else name[:45] + '...'

            lines.append(
                f'{display_name:<50} {"flux":<10} {flux_q[1,i]:>12.2f} '
                f'[{flux_q[0,i]:>11.2f}, {flux_q[2,i]:>11.2f}]'
            )
            lines.append(
                f'{"":<50} {"fwhm":<10} {fwhm_q[1,i]:>12.0f} '
                f'[{fwhm_q[0,i]:>11.0f}, {fwhm_q[2,i]:>11.0f}]'
            )
            lines.append(f'{"":<50} {"z":<10} {z_q[1,i]:>12.6f} ' f'[{z_q[0,i]:>11.6f}, {z_q[2,i]:>11.6f}]')
            lines.append('-' * 90)

    elif 'flux_all' in samples:
        # Fallback if no config provided
        flux_q = q(samples['flux_all'])
        fwhm_q = q(samples['fwhm_all'])
        z_q = q(samples['redshift_all'])
        n_lines = flux_q.shape[1]

        lines.append(f'\n{"Line":<12} {"Param":<8} {"P50":>12} {"[P16, P84]":>26}')
        lines.append('-' * 90)

        for i in range(n_lines):
            name = f'Line_{i}'
            lines.append(
                f'{name:<12} {"flux":<8} {flux_q[1,i]:>12.2f} '
                f'[{flux_q[0,i]:>11.2f}, {flux_q[2,i]:>11.2f}]'
            )
            lines.append(
                f'{"":<12} {"fwhm":<8} {fwhm_q[1,i]:>12.0f} '
                f'[{fwhm_q[0,i]:>11.0f}, {fwhm_q[2,i]:>11.0f}]'
            )
            lines.append(f'{"":<12} {"z":<8} {z_q[1,i]:>12.6f} ' f'[{z_q[0,i]:>11.6f}, {z_q[2,i]:>11.6f}]')
            lines.append('-' * 90)

    # Continuum parameters
    if 'mbb_amplitude' in samples:
        lines.append(f'\nContinuum (Modified Blackbody):')
        lines.append(f'{"Parameter":<50} {"P50":>12} {"[P16, P84]":>26}')
        lines.append('-' * 90)
        for key in ['mbb_amplitude', 'mbb_temperature', 'mbb_beta']:
            if key in samples:
                vals = q(samples[key])
                # Convert to float to handle 0-d arrays
                v = [float(vals[0]), float(vals[1]), float(vals[2])]
                if 'amplitude' in key:
                    fmt = '.2e'
                elif 'temp' in key:
                    fmt = '.0f'
                else:
                    fmt = '.2f'
                lines.append(f'{key.replace("mbb_", ""):<50} {v[1]:{fmt}} ' f'[{v[0]:{fmt}}, {v[2]:{fmt}}]')

    elif 'cont_angle' in samples:
        lines.append(f'\nContinuum (Linear):')
        lines.append(f'{"Parameter":<50} {"P50":>12} {"[P16, P84]":>26}')
        lines.append('-' * 90)
        angle_q = q(samples['cont_angle'])
        offset_q = q(samples['cont_offset'])
        n_reg = angle_q.shape[1] if angle_q.ndim > 1 else 1
        for i in range(n_reg):
            a = angle_q[:, i] if n_reg > 1 else angle_q
            o = offset_q[:, i] if n_reg > 1 else offset_q
            # Convert to float to handle 0-d arrays
            a_vals = [float(a[0]), float(a[1]), float(a[2])]
            o_vals = [float(o[0]), float(o[1]), float(o[2])]
            lines.append(
                f'{"Region " + str(i) + " - angle":<50} {a_vals[1]:>12.4f} '
                f'[{a_vals[0]:>11.4f}, {a_vals[2]:>11.4f}]'
            )
            lines.append(
                f'{"Region " + str(i) + " - offset":<50} {o_vals[1]:>12.2f} '
                f'[{o_vals[0]:>11.2f}, {o_vals[2]:>11.2f}]'
            )

    lines.append('=' * 90)
    return '\n'.join(lines)


def corner_plot(
    samples: dict,
    config: dict | None = None,
    max_lines: int = 20,
    continuum: bool = False,
    figsize: tuple = (12, 12),
    smooth: float = 1.0,
):
    """Create corner plot of fitted parameters.

    Parameters
    ----------
    samples : dict
        Samples dictionary from NPZ file.
    config : dict, optional
        Config dict for line names.
    max_lines : int
        Maximum number of lines to include (default 20).
        If more lines exist, only the brightest ones are shown.
    continuum : bool
        Include continuum parameters (default False).
    figsize : tuple
        Figure size (width, height) in inches.
    smooth : float
        Smoothing scale for 2D histograms (default 1.0).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Corner plot figure.
    """
    try:
        import corner as corner_pkg
    except ImportError:
        raise ImportError("corner package required for corner plots. " "Install with: pip install corner")

    # Collect samples and labels
    plot_samples = []
    labels = []

    # Line parameters
    if 'flux_all' in samples:
        flux_all = samples['flux_all']
        fwhm_all = samples['fwhm_all']
        z_all = samples['redshift_all']

        n_lines = flux_all.shape[1]

        # Get line names
        if config is not None:
            line_names = _get_line_names(config)
        else:
            line_names = [f'Line_{i}' for i in range(n_lines)]

        # If too many lines, select brightest ones
        if n_lines > max_lines:
            median_flux = np.median(flux_all, axis=0)
            brightest_idx = np.argsort(median_flux)[-max_lines:]
            line_indices = sorted(brightest_idx)
        else:
            line_indices = range(n_lines)

        # Track which fwhm and z samples have been added (to skip tied duplicates)
        seen_fwhm = []  # List of (samples_hash, label) tuples
        seen_z = []

        # Add line parameters
        for i in line_indices:
            name = line_names[i]
            # Shorten names for display
            if len(name) > 30:
                name = name[:27] + '...'

            # Flux is always unique per line
            plot_samples.append(flux_all[:, i])
            labels.append(f'{name}\nflux')

            # Check if fwhm is tied (duplicate of already-added fwhm)
            fwhm_i = fwhm_all[:, i]
            is_duplicate_fwhm = any(np.allclose(fwhm_i, seen) for seen, _ in seen_fwhm)
            if not is_duplicate_fwhm:
                plot_samples.append(fwhm_i)
                labels.append(f'{name}\nfwhm')
                seen_fwhm.append((fwhm_i, name))

            # Check if redshift is tied (duplicate of already-added z)
            z_i = z_all[:, i]
            is_duplicate_z = any(np.allclose(z_i, seen) for seen, _ in seen_z)
            if not is_duplicate_z:
                plot_samples.append(z_i)
                labels.append(f'{name}\nz')
                seen_z.append((z_i, name))

    # Continuum parameters - only if requested
    if continuum and 'mbb_amplitude' in samples:
        plot_samples.append(np.log10(samples['mbb_amplitude']))
        labels.append('log10(MBB\namplitude)')

        plot_samples.append(samples['mbb_temperature'])
        labels.append('MBB\ntemperature')

        if 'mbb_beta' in samples:
            plot_samples.append(samples['mbb_beta'])
            labels.append('MBB\nbeta')

    elif continuum and 'cont_angle' in samples:
        cont_angle = samples['cont_angle']
        cont_offset = samples['cont_offset']

        # Ensure arrays are at least 1D
        cont_angle = np.atleast_1d(cont_angle)
        cont_offset = np.atleast_1d(cont_offset)

        n_regions = cont_angle.shape[1] if cont_angle.ndim > 1 else 1

        for i in range(min(n_regions, 3)):  # Limit to 3 regions
            angle_data = cont_angle[:, i] if n_regions > 1 else cont_angle
            offset_data = cont_offset[:, i] if n_regions > 1 else cont_offset

            plot_samples.append(angle_data)
            labels.append(f'Cont R{i}\nangle')

            plot_samples.append(offset_data)
            labels.append(f'Cont R{i}\noffset')

    # Stack samples
    data = np.column_stack(plot_samples)

    # Create corner plot
    fig = corner_pkg.corner(
        data,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={'fontsize': 10},
        label_kwargs={'fontsize': 9},
        figsize=figsize,
        smooth=smooth,
    )

    return fig


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
        Model version to use ('v1' or 'v2'), default 'v2'

    Returns
    -------
    components : Dict[str, Dict[str, jnp.ndarray]]
        Dictionary of components for each spectrum.
        Keys are spectrum names.
        Values are dictionaries with keys 'lines' (n_pixels, n_lines) and 'continuum' (n_pixels,).
    config : dict
        The restricted configuration dictionary with 'Index' keys added, matching the model.
    """
    # Select model
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
    # This prevents shape mismatches if model_args (e.g. pixel grid) changed
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
        # Shape: (n_pixels, n_lines)
        lines = tr[f'{spec.name}_lines']['value']
        # Shape: (n_pixels,)
        wave = tr[f'{spec.name}_wave']['value']
        continuum = tr[f'{spec.name}_cont']['value']
        model = tr[f'{spec.name}_model']['value']

        # Apply flux scale so components match the model/data units
        flux_key = f'{spec.name}_flux'
        if flux_key in tr:
            flux_scale = tr[flux_key]['value']
            lines = lines * flux_scale
            continuum = continuum * flux_scale

        components[spec.name] = {'wave': wave, 'lines': lines, 'continuum': continuum, 'model': model}

    return components, config
