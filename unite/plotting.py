"""
Plotting functions for the results of the sampling
"""

# Standard library
import os
import re
import itertools
from typing import Tuple, Optional

# Numerical packages
import numpy as np

# Astropy packages
from astropy import units as u
from astropy.table import Table
from jax import numpy as jnp
from jax.scipy.special import logsumexp

# Plotting packages
from matplotlib import pyplot
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as mlines
from scipy.optimize import minimize
from unite.spectra import NIRSpecSpectrum


def get_ion_props(name: str) -> Tuple[str, str, Optional[str]]:
    """Get color, label, and linestyle for a given species name."""
    # Strip m<digits> at the end (e.g. [OIII]m1 -> [OIII])
    label = re.sub(r'm\d+$', '', name)
    n = label.lower()

    if 'feii' in n:
        return 'green', 'Fe II', '-'
    if 'heii' in n:
        return 'magenta', 'He II', '-'
    if 'hei' in n:
        return 'magenta', 'He I', '-'
    # HI check: Ha, Hb, Hg, Hd, Hepsilon, Pa, Pb, Pg
    if (n[0] in ['l', 'h', 'p']) and len(n) < 4 and n[1] in 'abgdeyi':
        return 'blue', 'H I', '-'

    # Other species: return None for linestyle to be assigned dynamically
    return 'orange', label, None


def plotRegion(
    ax: pyplot.Axes,
    wave: jnp.ndarray,
    flux: jnp.ndarray,
    err: jnp.ndarray,
    model_samples: jnp.ndarray,
    best_model_idx: int,
    region: Tuple[float, float],
    redshift: float,
    line_centers: jnp.ndarray,
    spectrum_name: str = "",
    plot_kwargs: dict | None = None,
    show_ylabel: bool = True,
    show_xlabel: bool = False,
    show_rest_labels: bool = True,
    resid_ax: pyplot.Axes | None = None,
    config: dict | None = None,
    components: dict | None = None,
) -> None:
    """
    Plot a single spectrum and model fit for a specific region.
    """
    plot_kwargs = plot_kwargs or {}
    show_lines = plot_kwargs.get('show_lines', True)

    mask = jnp.logical_and(wave > region[0], wave < region[1])

    ax.plot(wave[mask], flux[mask], color='k', ds='steps-mid')
    ax.errorbar(wave[mask], flux[mask], yerr=err[mask], fmt='none', color='k')

    # Check if model samples match the wave array
    m = None
    if model_samples.shape[1] == len(wave):
        for k in range(model_samples.shape[0]):
            ax.plot(
                wave[mask],
                model_samples[k][mask],
                color='#E20134',
                alpha=np.clip(5 / len(model_samples), 0.01, 1),
                ds='steps-mid',
            )
        m = model_samples[best_model_idx]
        ax.plot(wave[mask], m[mask], color='#A40122', alpha=1, lw=2, ds='steps-mid')
    else:
        print(
            f"Warning: Model samples shape {model_samples.shape} does not match wave shape {wave.shape}. Skipping samples plot."
        )

    # Add zero line for reference
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, lw=1)

    # Pre-compute styles for "Others" to ensure consistency
    other_styles = itertools.cycle(['-', '--', '-.', ':'])
    assigned_styles = {}

    if config is not None:
        for group in config['Groups'].values():
            for species in group['Species']:
                name = species['Name']
                _, label, linestyle = get_ion_props(name)
                if linestyle is None and label not in assigned_styles:
                    assigned_styles[label] = next(other_styles)

    # Plot Components
    if components is not None and config is not None:
        # Plot continuum
        if 'continuum' in components:
            cont = components['continuum']
            # ax.plot(wave[mask], cont[mask], color='gray', linestyle=':', alpha=0.8, label='Continuum')

        # Plot species
        if 'lines' in components:
            lines_arr = components['lines']
            plot_groups = {}

            for group in config['Groups'].values():
                for species in group['Species']:
                    name = species['Name']
                    color, label, linestyle = get_ion_props(name)
                    if linestyle is None:
                        linestyle = assigned_styles.get(label, '-')

                    # Collect indices for this species
                    indices = []
                    for line in species['Lines']:
                        if 'Index' in line:
                            indices.append(line['Index'])

                    if indices:
                        # Sum components
                        valid_indices = [i for i in indices if i < lines_arr.shape[1]]
                        if valid_indices:
                            species_flux = lines_arr[:, valid_indices].sum(axis=1)

                            ltype = species.get('LineType', '').lower()
                            # Group Broad and Absorption together
                            is_broad_complex = (
                                'broad' in ltype
                                or 'absorption' in ltype
                                or 'emission' in ltype
                                or 'exponential' in ltype
                                or 'lorentzian' in ltype
                            )

                            key = (label, is_broad_complex)
                            if key not in plot_groups:
                                plot_groups[key] = {
                                    'flux': jnp.zeros_like(species_flux),
                                    'color': color,
                                    'linestyle': linestyle,
                                }

                            plot_groups[key]['flux'] += species_flux

            # Plot the grouped components
            for (label, is_broad), data in plot_groups.items():
                s_flux = data['flux']
                color = data['color']
                linestyle = data.get('linestyle', '-')

                # Check visibility in mask
                if not jnp.any(mask):
                    continue

                # Check visibility in mask
                if jnp.max(s_flux[mask]) > 1e-10 or jnp.min(s_flux[mask]) < -1e-10:
                    ax.plot(wave[mask], s_flux[mask], color=color, linestyle=linestyle, alpha=0.8, lw=1.5)

    # Optional scales
    if "yscale" in plot_kwargs:
        yscale = plot_kwargs["yscale"]
        if isinstance(yscale, tuple):
            ax.set_yscale(yscale[0], **yscale[1])
        else:
            ax.set_yscale(yscale)
    if "xscale" in plot_kwargs:
        xscale = plot_kwargs["xscale"]
        if isinstance(xscale, tuple):
            ax.set_xscale(xscale[0], **xscale[1])
        else:
            ax.set_xscale(xscale)
    if "ylim" in plot_kwargs:
        y_bottom = plot_kwargs["ylim"][0]
        y_top = ax.get_ylim()[1] if len(plot_kwargs["ylim"]) == 1 else plot_kwargs["ylim"][1]
        ax.set_ylim(y_bottom, y_top)
    elif ax.get_yscale() in ['linear', 'symlog']:
        ax.set_ylim(bottom=0)

    if show_ylabel:
        ax.set_ylabel(r'$f_\lambda$ [10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]')

    if show_xlabel and resid_ax is None:
        ax.set_xlabel(r'$\lambda$ (Observed) [$\mu$m]')

    # Disperser text
    if spectrum_name:
        ax.text(0.02, 0.95, spectrum_name, transform=ax.transAxes, va='top', ha='left', fontsize=12)

    ax.set(xlim=region)

    rest_ax = ax.secondary_xaxis(
        'top', functions=(lambda x: x / (1 + redshift), lambda x: x * (1 + redshift))
    )
    if not show_rest_labels:
        rest_ax.set(xticklabels=[])
    else:
        rest_ax.set_xlabel(r'$\lambda$ (Rest) [$\mu$m]')

    ax.tick_params(axis='x', which='both', top=False)

    # Plot Lines
    legend_handles = {}
    if config is not None:
        for group in config['Groups'].values():
            for species in group['Species']:
                name = species['Name']
                color, label, linestyle = get_ion_props(name)
                if linestyle is None:
                    linestyle = assigned_styles.get(label, '-')

                for line in species['Lines']:
                    # Assume config unit is Angstrom, wave is Micron
                    unit_str = config.get('Unit', 'Angstrom')
                    unit = u.Unit(unit_str)

                    lam_rest = line['Wavelength'] * unit
                    lam_obs = lam_rest.to(u.micron).value * (1 + redshift)

                    if lam_obs >= region[0] and lam_obs <= region[1]:
                        if show_lines:
                            ax.axvline(lam_obs, color=color, linestyle='--', alpha=0.2)
                        if label not in legend_handles:
                            legend_handles[label] = mlines.Line2D(
                                [], [], color=color, linestyle=linestyle, label=label
                            )

        if legend_handles:
            ax.legend(
                handles=legend_handles.values(),
                loc='upper right',
                fontsize='small',
                frameon=True,
                framealpha=1,
            )

    else:
        for line in jnp.unique(line_centers):
            line = line * (1 + redshift)
            if line < region[0] or line > region[1]:
                continue
            ax.axvline(line, color='k', linestyle='--', alpha=0.5)

    # Minor ticks (3 ticks = 4 intervals)
    ax.minorticks_on()

    # Disable minor ticks for symlog
    yscale = plot_kwargs.get("yscale", "linear")
    is_symlog = (isinstance(yscale, tuple) and yscale[0] == 'symlog') or (yscale == 'symlog')

    if not is_symlog:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    rest_ax.minorticks_on()
    rest_ax.xaxis.set_minor_locator(AutoMinorLocator())

    tick_labelsize = plot_kwargs.get("tick_labelsize", 12)
    tick_length = plot_kwargs.get("tick_length", 6)
    tick_width = plot_kwargs.get("tick_width", 1.1)

    ax.tick_params(
        axis='both', which='major', labelsize=tick_labelsize, length=tick_length, width=tick_width
    )
    ax.tick_params(axis='both', which='minor', length=tick_length / 2, width=tick_width)

    rest_ax.tick_params(
        axis='x',
        which='major',
        labelsize=max(tick_labelsize - 1, 8),
        length=tick_length * 0.6,
        width=tick_width * 0.8,
    )
    rest_ax.tick_params(axis='x', which='minor', length=tick_length * 0.3, width=tick_width * 0.8)

    if resid_ax is not None and m is not None:
        resid = (flux[mask] - m[mask]) / err[mask]
        resid_ax.plot(wave[mask], resid, color='k', ds='steps-mid')
        resid_ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        resid_ax.set_ylabel(r'$\chi$')
        resid_ax.set_xlim(region)
        resid_ax.set_ylim(-5.5, 5.5)

        # Calculate Chi2 and WAIC
        chi2 = jnp.sum(resid**2)

        waic_str = ""
        if model_samples.shape[1] == len(wave):
            ms = model_samples[:, mask]
            y = flux[mask]
            e = err[mask]

            # Log-likelihood: -0.5 * ((y - mu)/e)^2 - 0.5 * log(2*pi*e^2)
            LL = -0.5 * ((y - ms) / e) ** 2 - 0.5 * jnp.log(2 * jnp.pi * e**2)

            # lppd = sum(log(mean(exp(LL))))
            S = ms.shape[0]
            lppd = jnp.sum(logsumexp(LL, axis=0) - jnp.log(S))

            # p_waic = sum(var(LL))
            p_waic = jnp.sum(jnp.var(LL, axis=0))

            waic = -2 * (lppd - p_waic)
            waic_str = f"\nWAIC = {waic:.1f}"

        resid_ax.text(
            0.02,
            0.92,
            f"$\chi^2$ = {chi2:.1f}{waic_str}",
            transform=resid_ax.transAxes,
            va='top',
            ha='left',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
        )

        # Hide x labels on main ax
        ax.tick_params(labelbottom=False)

        resid_ax.minorticks_on()
        resid_ax.xaxis.set_minor_locator(AutoMinorLocator())
        resid_ax.yaxis.set_minor_locator(AutoMinorLocator())

        resid_ax.tick_params(
            axis='both', which='major', labelsize=tick_labelsize, length=tick_length, width=tick_width
        )
        resid_ax.tick_params(axis='both', which='minor', length=tick_length / 2, width=tick_width)

        if show_xlabel:
            resid_ax.set_xlabel(r'$\lambda$ (Observed) [$\mu$m]')


def plotResults(
    config: list,
    rows: Table,
    output_dir: str,
    samples: dict | None = None,
    model_args: tuple | None = None,
    plot_kwargs: dict | None = None,
    components: dict | None = None,
) -> Tuple[pyplot.Figure, dict]:
    """
    Plot the results of the sampling.

    Parameters
    ----------
    config: list
        Configuration list
    rows : Table
        Table of the rows
    model_args : tuple
        Arguments for the model
    samples : dict
        Samples from the MCMC. If None, will try to load from disk.
    output_dir : str
        Directory to save the plots
    plot_kwargs : dict, optional
        Per-axis plotting options, e.g.
        {
            "tick_labelsize": 12,
            "tick_length": 6,
            "tick_width": 1.2,
            "yscale": ("symlog", {"linthresh": 1e-18}),  # or "log"
            "xscale": "linear",
            "show_lines": True
        }
    components : dict, optional
        Pre-computed components from get_components_fit

    Returns
    -------
    fig : pyplot.Figure
        The figure object
    data : dict
        Dictionary containing samples, components, and model data
    """
    plot_kwargs = plot_kwargs or {}
    cname = '_' + config['Name'] if config['Name'] else ''

    # Default rescale_errors to False if not provided (or add it to args)
    rescale_errors = False
    from unite.fitting import NIRSpecModelArgs

    # Respect provided model_args; only rebuild if missing
    if model_args is None:
        _, model_args = NIRSpecModelArgs(config, rows=rows, rescale_errors=rescale_errors)

    # Respect provided samples; only load from disk if missing
    if samples is None:
        savename = f'{output_dir}/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}'
        samples = np.load(f'{savename}_full.npz')

    # Compute components if not provided
    if components is None:
        from unite.fitting import get_components_fit

        components, config = get_components_fit(config, model_args, samples)

    os.makedirs(f'{output_dir}/Plots/', exist_ok=True)

    spectra, _, _, line_centers, _, fit_regions, _ = model_args
    Nspec, Nregs = len(spectra.spectra), len(fit_regions)

    # Increased height slightly to accommodate residuals
    figsize = (7.5 * Nregs, 7.5 * Nspec)
    fig = pyplot.figure(figsize=figsize, constrained_layout=True)

    # Create outer grid for spectra (rows) and regions (cols)
    outer_grid = fig.add_gridspec(Nspec, Nregs, hspace=0.1, wspace=0.1)

    best_model_idx = samples['logP'].argmax()

    # Initialize data return structure
    data = {'samples': samples, 'components': components, 'best_model_idx': best_model_idx, 'spectra': {}}

    for i, spectrum in enumerate(spectra.spectra):
        _, wave, _, flux, err = spectrum()
        model_samples = samples[f'{spectrum.name}_model']

        # Store spectrum specific data
        data['spectra'][spectrum.name] = {
            'wave': wave,
            'flux': flux,
            'err': err,
            'model_samples': model_samples,
            'best_fit': model_samples[best_model_idx],
        }

        # Symlog default
        current_plot_kwargs = plot_kwargs.copy()
        if "yscale" not in current_plot_kwargs:
            med = np.nanmedian(flux)
            if med <= 0 or np.isnan(med):
                med = 1.0
            #            current_plot_kwargs["yscale"] = ("symlog", {"linthresh": med})
            current_plot_kwargs["yscale"] = "linear"

        for j in range(Nregs):
            cont_reg = fit_regions[j]

            # Create inner grid for main plot + residual
            inner_grid = outer_grid[i, j].subgridspec(2, 1, height_ratios=[4, 1], hspace=0)
            ax = fig.add_subplot(inner_grid[0])
            resid_ax = fig.add_subplot(inner_grid[1], sharex=ax)

            plotRegion(
                ax=ax,
                wave=wave,
                flux=flux,
                err=err,
                model_samples=model_samples,
                best_model_idx=best_model_idx,
                region=cont_reg,
                redshift=spectra.redshift_initial,
                line_centers=line_centers,
                spectrum_name=spectrum.name,
                plot_kwargs=current_plot_kwargs,
                show_ylabel=False,
                show_xlabel=(i == Nspec - 1),
                show_rest_labels=(i == 0),
                resid_ax=resid_ax,
                config=config,
                components=components.get(spectrum.name),
            )

            # Handle x-labels
            if i != Nspec - 1:
                resid_ax.tick_params(labelbottom=False)

    fig.supylabel(rf'$f_\lambda$ [{spectrum.fλ_unit.to_string(format="latex", fraction=False)}]')
    fig.supxlabel(
        rf'$\lambda$ (Observed) [{spectrum.λ_unit.to_string(format="latex", fraction=False)}]',
        y=-0.01,
        va='center',
        fontsize='medium',
    )
    fig.suptitle(
        rf'$\lambda$ (Rest) [{spectrum.λ_unit:latex_inline}]', y=1.015, va='center', fontsize='medium'
    )
    fig.text(
        0.5,
        1.05,
        f'{rows[0]["srcid"]} ({rows[0]["root"]}): $z = {spectrum.redshift_initial:.3f}$',
        ha='center',
        va='center',
        fontsize='large',
    )

    fig.savefig(
        os.path.join(f'{output_dir}/Plots', f'{rows[0]["root"]}-{rows[0]["srcid"]}{cname}_fit.png'), dpi=300
    )
    pyplot.close(fig)

    return fig, data


from copy import deepcopy


def plotRegionSingle(
    config: dict,
    rows: Table,
    output_dir: str,
    spec: NIRSpecSpectrum,
    region: Tuple[float, float],
    model_args: tuple | None = None,
    samples: dict | None = None,
    ax: pyplot.Axes = None,
    plot_kwargs: dict | None = None,
    resid_ax: pyplot.Axes = None,
    components: dict | None = None,
    rescale_errors=False,
) -> None:
    """
    Driver to plot a single region for a single spectrum.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    spec : NIRSpecSpectrum
        The spectrum object to plot
    region : Tuple[float, float]
        The wavelength region (min, max)
    model_args : tuple
        Model arguments (contains line centers, etc.)
    samples : dict
        MCMC samples
    ax : pyplot.Axes, optional
        Axes to plot on. If None, creates a new figure.
    plot_kwargs : dict, optional
        Plotting keyword arguments
    resid_ax : pyplot.Axes, optional
        Axes to plot residuals on.
    components : dict, optional
        Pre-computed components
    """
    if ax is None:
        fig = pyplot.figure(figsize=(10, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0)
        ax = fig.add_subplot(gs[0])
        resid_ax = fig.add_subplot(gs[1], sharex=ax)

    if model_args is None:
        from unite.fitting import NIRSpecModelArgs

        _, model_args = NIRSpecModelArgs(config, rows=rows, rescale_errors=rescale_errors)

    # Respect provided samples; only load from disk if missing
    if samples is None:
        cname = '_' + config['Name'] if config['Name'] else ''
        savename = f'{output_dir}/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}'
        samples = np.load(f'{savename}_full.npz')

    # Compute components if not provided
    if components is None:
        from unite.fitting import get_components_fit

        components, config = get_components_fit(config, model_args, samples)

    # Unpack necessary info from model_args
    _, _, _, line_centers, _, fit_regions, _ = model_args

    # Get spectrum data
    spec_r = deepcopy(spec)
    spec_r.restrict(fit_regions)
    _, wave, _, flux, err = spec_r()

    # Get model samples for this spectrum
    model_samples = samples[f'{spec.name}_model']
    best_model_idx = samples['logP'].argmax()

    plotRegion(
        ax=ax,
        wave=wave,
        flux=flux,
        err=err,
        model_samples=model_samples,
        best_model_idx=best_model_idx,
        region=region,
        redshift=spec.redshift_initial,
        line_centers=line_centers,
        spectrum_name=spec.name,
        plot_kwargs=plot_kwargs,
        show_ylabel=True,
        show_xlabel=True,
        show_rest_labels=True,
        resid_ax=resid_ax,
        config=config,
        components=components.get(spec.name),
    )
    return ax


def plotLines(ax, config, model_args) -> None:
    """
    Plot the lines of the spectra.

    Parameters
    ----------
    ax : Axes
        Axes to plot the lines
    config: list
        Configuration list
    model_args : tuple
        Arguments for the model

    Returns
    -------
    None

    """

    # Unpack model arguements
    spectra, _, _, _, _, _, _ = model_args
    oneplusz = 1 + spectra.redshift_initial

    # Get axis xlim and ylim
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Iterate over configuration
    names, centers = [], []
    for group in config['Groups'].values():
        for species in group['Species']:
            for line in species['Lines']:
                # Get the line center
                line_center = (
                    (line['Wavelength'] * oneplusz * u.Unit(config['Unit'])).to(spectra.λ_unit).value
                )

                # Check if line is in the axis limits
                if line_center < xlim[0] or line_center > xlim[1]:
                    continue

                # Append to names and centers
                names.append(species['Name'])
                centers.append(line_center)

    # Sort by wavelength
    names, centers = zip(*sorted(zip(names, centers), key=lambda x: x[1]))

    # Get name centers
    x0 = np.linspace(xlim[0], xlim[1], len(names) + 2)[1:-1]
    namecenters = minimize(
        logbarrier,
        x0,
        args=(xlim, centers, 1000),
        method='Nelder-Mead',
        options={'adaptive': True, 'maxiter': len(names) * 750},
    ).x

    # Plot the lines
    for name, ncenter, center in zip(names, namecenters, centers):
        ax.plot(
            [center, center, ncenter, ncenter],
            [ylim[0], ylim[1] * 0.92, ylim[1] * 0.93, ylim[1] * 0.95],
            color='k',
            linestyle='--',
            alpha=0.5,
        )

        ax.text(ncenter, ylim[1] * 0.96, name, va='center', ha='center', fontsize=15)

    # Set the axis limits
    ax.set(xlim=xlim, ylim=ylim)


# Log barrier constraints
def logbarrier(x, xlim, linelocs, norm):
    y = np.concatenate([[xlim[0]], x, [xlim[1]]])
    return np.square(x - linelocs).sum() - np.log(y[1:] - y[:-1]).sum() / norm


def plotFullSpectrum(
    config: dict,
    rows: Table,
    output_dir: str,
    samples: dict | None = None,
    model_args: tuple | None = None,
    wavelength_range: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (12, 5),
    rescale_errors: bool = False,
) -> Tuple[pyplot.Figure, pyplot.Axes]:
    """
    Plot the full spectrum with the model fit.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    rows : Table
        Table of the rows
    output_dir : str
        Directory to load results from
    samples : dict, optional
        Samples from the MCMC. If None, will try to load from disk.
    model_args : tuple, optional
        Arguments for the model. If None, will be computed.
    wavelength_range : Tuple[float, float], optional
        Wavelength range to plot (min, max) in microns
    figsize : Tuple[float, float]
        Figure size
    rescale_errors : bool
        Whether errors were rescaled during fitting

    Returns
    -------
    fig : pyplot.Figure
        The figure object
    ax : pyplot.Axes
        The axes object
    """
    cname = '_' + config['Name'] if config['Name'] else ''

    from unite.fitting import NIRSpecModelArgs

    if model_args is None:
        _, model_args = NIRSpecModelArgs(config, rows=rows, rescale_errors=rescale_errors)

    if samples is None:
        savename = f'{output_dir}/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}'
        samples = np.load(f'{savename}_full.npz')

    spectra, _, _, line_centers, _, fit_regions, _ = model_args
    best_model_idx = samples['logP'].argmax()

    fig, ax = pyplot.subplots(figsize=figsize)

    for spectrum in spectra.spectra:
        _, wave, _, flux, err = spectrum()
        model_samples = samples[f'{spectrum.name}_model']
        best_model = model_samples[best_model_idx]

        # Plot data
        ax.plot(wave, flux, color='k', ds='steps-mid', label='Data' if spectrum == spectra.spectra[0] else None)
        ax.fill_between(wave, flux - err, flux + err, alpha=0.2, color='gray', step='mid')

        # Plot model samples
        for k in range(model_samples.shape[0]):
            ax.plot(
                wave,
                model_samples[k],
                color='#E20134',
                alpha=np.clip(5 / len(model_samples), 0.01, 0.3),
                ds='steps-mid',
            )

        # Plot best fit
        ax.plot(wave, best_model, color='#A40122', alpha=1, lw=1.5, ds='steps-mid',
                label='Model' if spectrum == spectra.spectra[0] else None)

    # Mark fitting regions
    for region in fit_regions:
        ax.axvspan(region[0], region[1], alpha=0.05, color='blue')

    # Mark line centers
    oneplusz = 1 + spectra.redshift_initial
    for lc in jnp.unique(line_centers):
        ax.axvline(lc * oneplusz, color='gray', linestyle='--', alpha=0.3, lw=0.5)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, lw=1)

    if wavelength_range is not None:
        ax.set_xlim(wavelength_range)

    ax.set_xlabel(r'$\lambda$ (Observed) [$\mu$m]')
    ax.set_ylabel(r'$f_\lambda$ [10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]')
    ax.legend(loc='upper right')
    ax.set_title(f'{rows[0]["srcid"]} ({rows[0]["root"]}): $z = {spectra.redshift_initial:.3f}$')

    return fig, ax
