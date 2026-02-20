"""
Fitting functions for spectral data
"""

# Standard library
import re
import os
from copy import deepcopy
from dataclasses import dataclass, field

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


@dataclass
class FitResult:
    """Result container for a unite spectral fit."""

    samples: dict
    best_fit: dict
    extras: dict
    config: dict
    model_args: tuple
    output_dir: str = ''

    # Derived attributes (computed in __post_init__)
    n_samples: int = field(init=False, repr=False)
    spectra_names: list[str] = field(init=False, repr=False)
    line_names: list[str] = field(init=False, repr=False)
    continuum_type: str = field(init=False, repr=False)
    fitting_mode: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_samples = len(self.samples['logP'])
        spectra = self.model_args[0]
        self.spectra_names = [s.name for s in spectra.spectra]
        self.line_names = _build_line_names(self.config)
        self.continuum_type = self.config.get('continuum', {}).get('type', 'linear')
        self.fitting_mode = str(_infer_fitting_mode(self.config).value)

    def __getitem__(self, key: str):
        """Dict-style access for backward compatibility."""
        return getattr(self, key)

    def __repr__(self) -> str:
        name = self.config.get('Name', '')
        waic = self.extras.get('WAIC')
        waic_str = f', WAIC={waic:.2f}' if waic is not None else ''
        return (
            f"FitResult('{name}', {self.n_samples} samples, "
            f"{len(self.spectra_names)} spectra, {len(self.line_names)} lines{waic_str})"
        )

    def summary(self) -> None:
        """Print a compact summary of the fit result."""
        name = self.config.get('Name', '')
        spectra = self.model_args[0]
        _type_short = {'emission': 'em', 'absorption': 'abs'}
        c_kms = 299_792.458

        def _abbrev_keys(d: dict, n: int = 6) -> str:
            keys = list(d.keys())
            if len(keys) <= n:
                return repr(keys)
            return repr(keys[:n])[:-1] + ', ...]'

        out: list[str] = []

        # Run name, then fit info on next line
        out.append(f'Run: {name}')
        info_parts = [f'{self.fitting_mode} mode', f'{self.n_samples} samples']
        if 'WAIC' in self.extras:
            info_parts.append(f'WAIC={self.extras["WAIC"]:.2f}')
        if 'logZ' in self.extras:
            info_parts.append(f'logZ={self.extras["logZ"]:.2f}')
        out.append('   '.join(info_parts))
        out.append('')

        # Spectra (vertical, offset after label)
        sw = max(len(s.name) for s in spectra.spectra) + 2
        prefix = 'Spectra:  '
        indent = ' ' * len(prefix)
        for i, spec in enumerate(spectra.spectra):
            p = prefix if i == 0 else indent
            w_lo = float(spec.wave.min()) * spectra.λ_unit.to(u.um)
            w_hi = float(spec.wave.max()) * spectra.λ_unit.to(u.um)
            out.append(f'{p}{spec.name:<{sw}}{w_lo:.2f}\u2013{w_hi:.2f} \u00b5m')

        # Lines (skip if not fitted, e.g. continuum-only mode)
        show_lines = (
            self.fitting_mode != 'continuum'
            and self.line_names
            and 'redshift_all' in self.samples
        )
        if show_lines:
            z_all = np.array(self.samples['redshift_all'])
            flux_all = np.array(self.samples['flux_all'])
            fwhm_all = np.array(self.samples['fwhm_all'])
            n_lines = len(self.line_names)

            # Labels with species: "HI 6563 abs"
            labels: list[str] = []
            for group in self.config['Groups'].values():
                for species in group['Species']:
                    st = _type_short.get(species['LineType'], species['LineType'][:3])
                    for line in species['Lines']:
                        labels.append(f'{species["Name"]} {line["Wavelength"]} {st}')
            lw = max(len(l) for l in labels)

            # Reference line (brightest = highest median flux)
            f_meds = [float(np.median(flux_all[:, i])) for i in range(n_lines)]
            ref_idx = int(np.argmax(f_meds))
            z_ref = float(np.median(z_all[:, ref_idx]))

            # Pre-compute all row values for column alignment
            sep = '  '
            row_data: list[tuple] = []
            for i, label in enumerate(labels):
                z_q = np.percentile(z_all[:, i], [16, 50, 84])
                f_q = np.percentile(flux_all[:, i], [16, 50, 84])
                fw_q = np.percentile(fwhm_all[:, i], [16, 50, 84])
                dv = c_kms * (z_q[1] - z_ref) / (1 + z_ref)
                row_data.append((label, z_q, f_q, fw_q, dv, i == ref_idx))

            # Find max widths for value parts (align decimals)
            f_vals = [f'{r[2][1]:.2f}' for r in row_data]
            f_los = [f'{r[2][1]-r[2][0]:.2f}' for r in row_data]
            f_his = [f'{r[2][2]-r[2][1]:.2f}' for r in row_data]
            fw_vals = [f'{r[3][1]:.0f}' for r in row_data]
            fw_los = [f'{r[3][1]-r[3][0]:.0f}' for r in row_data]
            fw_his = [f'{r[3][2]-r[3][1]:.0f}' for r in row_data]
            wf = max(len(s) for s in f_vals)
            wfl = max(len(s) for s in f_los)
            wfh = max(len(s) for s in f_his)
            wfw = max(len(s) for s in fw_vals)
            wfwl = max(len(s) for s in fw_los)
            wfwh = max(len(s) for s in fw_his)

            # Build aligned strings
            rows: list[tuple[str, str, str, str, str]] = []
            for j, (label, z_q, f_q, fw_q, dv, is_ref) in enumerate(row_data):
                dv_str = '(0)' if is_ref else f'({dv:+.0f})'
                flux_str = f'{f_vals[j]:>{wf}} (-{f_los[j]:>{wfl}} +{f_his[j]:>{wfh}})'
                fwhm_str = f'{fw_vals[j]:>{wfw}} (-{fw_los[j]:>{wfwl}} +{fw_his[j]:>{wfwh}})'
                rows.append((label, f'{z_q[1]:.4f}', dv_str, flux_str, fwhm_str))

            # Column widths (with minimum for headers)
            cw = [lw, max(6, *(len(r[1]) for r in rows)),
                  max(7, *(len(r[2]) for r in rows)),
                  max(len('flux (1e-20 cgs)'), *(len(r[3]) for r in rows)),
                  max(len('fwhm (km/s)'), *(len(r[4]) for r in rows))]
            hdrs = ['Lines:', 'z', '(km/s)', 'flux (1e-20 cgs)', 'fwhm (km/s)']

            out.append('')
            out.append(sep.join(
                f'{hdrs[0]:<{cw[0]}}' if j == 0 else f'{hdrs[j]:>{cw[j]}}'
                for j in range(5)
            ))
            for label, z_s, dv_s, f_s, fw_s in rows:
                out.append(sep.join([
                    f'{label:<{cw[0]}}', f'{z_s:>{cw[1]}}', f'{dv_s:>{cw[2]}}',
                    f'{f_s:>{cw[3]}}', f'{fw_s:>{cw[4]}}',
                ]))

        # Continuum
        out.append('')
        cont_cfg = self.config.get('continuum', {})
        cont_desc = self.continuum_type
        cont_details = {k: v for k, v in cont_cfg.items() if k != 'type'}
        if cont_details:
            cont_desc += ' (' + ', '.join(f'{k}={v}' for k, v in cont_details.items()) + ')'
        out.append(f'Continuum: {cont_desc}')

        # Show continuum best-fit parameters (always for blackbody, else full/continuum only)
        _bb_types = ('blackbody', 'modified_blackbody', 'attenuated_blackbody')
        show_cont_params = self.fitting_mode in ('full', 'continuum') or self.continuum_type in _bb_types
        if show_cont_params:
            _cont_prefixes = ('cont_', 'cheb_', 'bb_', 'bsp_', 'bern_')
            _cont_exclude = {'cont_center'}
            cont_keys = [
                k for k in self.samples
                if any(k.startswith(p) for p in _cont_prefixes)
                and k not in _cont_exclude
            ]
            show_errors = self.continuum_type in _bb_types
            for k in cont_keys:
                arr = np.array(self.samples[k])
                if arr.ndim == 1:
                    q = np.percentile(arr, [16, 50, 84])
                    if show_errors:
                        out.append(f'  {k} = {q[1]:.4g} (-{q[1]-q[0]:.4g} +{q[2]-q[1]:.4g})')
                    else:
                        out.append(f'  {k} = {q[1]:.4g}')
                else:
                    qs = np.percentile(arr, [16, 50, 84], axis=0)
                    vals = '  '.join(f'{qs[1,j]:.4g}' for j in range(qs.shape[1]))
                    out.append(f'  {k} = [{vals}]')

            # Show fit regions
            if 'fit_regions' in self.samples:
                regs = np.asarray(self.samples['fit_regions'])
                if len(regs) > 1:
                    reg_strs = ', '.join(f'{lo:.3f}\u2013{hi:.3f}' for lo, hi in regs)
                    out.append(f'  fit_regions = [{reg_strs}]')

        # Contents with key names
        out.append('')
        out.append(f'.samples    {self.n_samples} \u00d7 {len(self.samples)} {_abbrev_keys(self.samples)}')
        out.append(f'.best_fit   {len(self.best_fit)} keys {_abbrev_keys(self.best_fit)}')
        extras_parts = [
            f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}'
            for k, v in self.extras.items()
        ]
        if 'logP' in self.samples:
            extras_parts.append(f'max_logP={float(self.samples["logP"].max()):.2f}')
        out.append(f'.extras     {{{", ".join(extras_parts)}}}')
        out.append(f'.config     {list(self.config.keys())}')
        out.append(f'.model_args (spectra, matrices, linetypes, centers, estimates, regions, continuum)')
        if self.output_dir:
            cname = '_' + name if name else ''
            out.append('')
            out.append(f'Saved: {self.output_dir}/Results/*{cname}_summary.csv, {self.output_dir}/Plots/')

        # Thin outer box
        W = max(len(l) for l in out)
        print('\u250c' + '\u2500' * (W + 2) + '\u2510')
        for line in out:
            print('\u2502 ' + line.ljust(W) + ' \u2502')
        print('\u2514' + '\u2500' * (W + 2) + '\u2518')


def _infer_fitting_mode(config: dict) -> FittingMode:
    """Infer fitting mode from config.

    Explicit ``config['fitting_mode']`` takes precedence.
    Otherwise: linear continuum → ``LINES``, anything else → ``FULL``.
    """
    if 'fitting_mode' in config:
        return FittingMode(config['fitting_mode'])
    cont_type = config.get('continuum', {}).get('type', 'linear').lower()
    if cont_type in ('linear', 'chebyshev'):
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
    rng_seed: int = 0,
    nuts_dense_mass: bool = True,
    nuts_target_accept_prob: float = 0.8,
    nuts_max_tree_depth: int = 10,
    flux_reg: float = 0.0,
) -> 'FitResult':
    """Run a full spectral fit (model setup, MCMC, save, plot).

    Parameters
    ----------
    flux_reg : float
        Flux regularization strength (sigma). Adds a Gaussian penalty
        ``-0.5 * sum(flux^2) / flux_reg^2`` to the log-likelihood,
        discouraging degenerate large emission/absorption pairs.
        Set to 0 (default) to disable.

    Returns
    -------
    FitResult
        Result container with ``samples``, ``best_fit``, ``extras``,
        ``config``, ``model_args`` attributes (also supports dict-style access).
    """
    # Get the model arguments
    config, model_args = NIRSpecModelArgs(
        config, rows=rows, spectra=spectra, rescale_errors=rescale_errors, model_version=model_version,
        flux_reg=flux_reg,
    )

    # Get rows if not provided (for saveResults)
    if rows is None:
        rows = model_args[0].rows

    # Get the random key
    rng_key = random.PRNGKey(rng_seed)

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
                dense_mass=nuts_dense_mass,
                target_accept_prob=nuts_target_accept_prob,
                max_tree_depth=nuts_max_tree_depth,
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

    # Best-fit parameters (max-likelihood sample)
    best_idx = int(samples['logP'].argmax())
    n_samples = len(samples['logP'])
    best_fit = {k: v[best_idx] if v.shape[0] == n_samples else v for k, v in samples.items()}

    return FitResult(
        samples=samples,
        best_fit=best_fit,
        extras=extras,
        config=config,
        model_args=model_args,
        output_dir=output_directory,
    )


def NIRSpecModelArgs(
    config: dict,
    rows: Table | None = None,
    spectra: NIRSpecSpectra | None = None,
    rescale_errors=True,
    model_version: str = 'v2',
    flux_reg: float = 0.0,
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

    # Register custom line type(s) from config
    if 'Linetype' in config:
        lt = config['Linetype']
        entries = lt if isinstance(lt, list) else [lt]
        for entry in entries:
            defaults.register_linetype(
                entry['name'], entry['flux'], entry['redshift'], entry['fwhm']
            )

    # Restrict config to what we have coverage of
    config = utils.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Generate Parameter Matrices
    matrices, linetypes_all = parameters.configToMatrices(config)

    # Determine fitting mode: explicit config value, or infer from continuum type
    fitting_mode = _infer_fitting_mode(config)

    # Compute fitting regions (observed-frame)
    fit_regions = initial.compute_fit_regions(config, spectra, mode=fitting_mode)

    # Compute per-spectrum line mask on full (unrestricted) arrays
    for spectrum in spectra.spectra:
        spectrum.compute_line_mask(config)

    # Restrict spectra to fitting regions and rescale errorbars
    spectra.restrictAndRescale(config, fit_regions, rescale_errors=rescale_errors)

    # Continuum height guesses (uses stored line_mask)
    cont_guesses = initial.continuumHeightGuesses(fit_regions, spectra)

    # Compute Line Centers and Equalized estimates
    line_centers, line_estimates_eq = initial.linesFluxesGuess(config, spectra, fit_regions, cont_guesses)

    # In CONTINUUM mode, remove emission-line pixels using stored mask
    if fitting_mode == FittingMode.CONTINUUM:
        for spectrum in spectra.spectra:
            mask = spectrum.line_mask
            for key in ['wave', 'low', 'high', 'flux', 'err', 'valid', 'line_mask']:
                setattr(spectrum, key, getattr(spectrum, key)[mask])
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
        continuum_model.flux_reg = flux_reg
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
    dense_mass: bool = True,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
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

    # MCMC knobs: can trade robustness for speed when needed.
    kernel = infer.NUTS(
        model_fn,
        init_strategy=init_strategy,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
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

    # Compute the log likelihood (filter out scalar factor sites)
    logLs = infer.util.log_likelihood(model_fn, samples, *model_args)
    for k, v in logLs.items():
        samples[k] = v
    logL = np.hstack([p for p in logLs.values() if p.ndim == 2])  # Likelihood Matrix
    samples['logL'] = logL.sum(1)

    # Compute the log density
    logP = vmap(lambda s: infer.util.log_density(model_fn, model_args, {}, s)[0])(samples)
    samples['logP'] = np.array(logP)

    return logL


def _build_line_names(config: dict) -> list[str]:
    """Build unique line names from config groups.

    Returns
    -------
    list[str]
        Unique line names like ``'HI_Ha_emission_6563'``.
    """
    names = [
        (
            f'{group_name}_{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}'
            if group_name
            else f'{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}'
        )
        for group_name, group in config['Groups'].items()
        for species in group['Species']
        for line in species['Lines']
    ]
    # Make duplicates unique by appending index suffix
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f'{name}_{seen[name]}')
        else:
            seen[name] = 0
            unique.append(name)
    return unique


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

    # Create outputs: calibration params (lsf_scale, per-spectrum flux/offset) + logL/logP
    calib_keys = ['lsf_scale']
    for spectrum in spectra.spectra:
        calib_keys.extend([f'{spectrum.name}_flux', f'{spectrum.name}_offset'])
    colnames = [n for n in calib_keys + ['logL', 'logP'] if n in samples]
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
    line_names = _build_line_names(config)

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
