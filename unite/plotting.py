"""
Plotting functions for the results of the sampling
"""

# Standard library
import os
from typing import Tuple

# Numerical packages
import numpy as np

# Astropy packages
from astropy import units as u
from astropy.table import Table
from jax import numpy as jnp

# Plotting packages
from matplotlib import pyplot
from scipy.optimize import minimize


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
    show_rest_labels: bool = True,
    resid_ax: pyplot.Axes | None = None,
) -> None:
    """
    Plot a single spectrum and model fit for a specific region.
    """
    plot_kwargs = plot_kwargs or {}

    mask = jnp.logical_and(wave > region[0], wave < region[1])

    ax.plot(wave[mask], flux[mask], color='k', ds='steps-mid')
    ax.errorbar(wave[mask], flux[mask], yerr=err[mask], fmt='none', color='k')

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

    if show_ylabel:
        ax.set(ylabel=f'{spectrum_name}')
    ax.set(xlim=region)

    rest_ax = ax.secondary_xaxis(
        'top', functions=(lambda x: x / (1 + redshift), lambda x: x * (1 + redshift))
    )
    if not show_rest_labels:
        rest_ax.set(xticklabels=[])
    ax.tick_params(axis='x', which='both', top=False)

    for line in jnp.unique(line_centers):
        line = line * (1 + redshift)
        if line < region[0] or line > region[1]:
            continue
        ax.axvline(line, color='k', linestyle='--', alpha=0.5)

    tick_labelsize = plot_kwargs.get("tick_labelsize", 12)
    tick_length = plot_kwargs.get("tick_length", 6)
    tick_width = plot_kwargs.get("tick_width", 1.1)

    ax.tick_params(axis='both', which='both', labelsize=tick_labelsize, length=tick_length, width=tick_width)
    rest_ax.tick_params(
        axis='x',
        which='both',
        labelsize=max(tick_labelsize - 1, 8),
        length=tick_length * 0.6,
        width=tick_width * 0.8,
    )

    if resid_ax is not None:
        resid = (flux[mask] - m[mask]) / err[mask]
        resid_ax.plot(wave[mask], resid, color='k', ds='steps-mid')
        resid_ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        resid_ax.set_ylabel(r'$\chi$')
        resid_ax.set_xlim(region)

        # Hide x labels on main ax
        ax.tick_params(labelbottom=False)

        resid_ax.tick_params(
            axis='both', which='both', labelsize=tick_labelsize, length=tick_length, width=tick_width
        )


def plotResults(
    config: list,
    rows: Table,
    model_args: tuple,
    samples: dict,
    output_dir: str,
    plot_kwargs: dict | None = None,
) -> None:
    """
    Plot the results of the sampling.

    Parameters
    ----------
    output_dir : str
        Directory to save the plots
    config: list
        Configuration list
    rows : Table
        Table of the rows
    model_args : tuple
        Arguments for the model
    samples : dict
        Samples from the MCMC
    plot_kwargs : dict, optional
        Per-axis plotting options, e.g.
        {
            "tick_labelsize": 12,
            "tick_length": 6,
            "tick_width": 1.2,
            "yscale": ("symlog", {"linthresh": 1e-18}),  # or "log"
            "xscale": "linear"
        }
    """
    plot_kwargs = plot_kwargs or {}
    cname = '_' + config['Name'] if config['Name'] else ''

    os.makedirs(f'{output_dir}/Plots/', exist_ok=True)

    spectra, _, _, line_centers, _, cont_regs, _ = model_args
    Nspec, Nregs = len(spectra.spectra), len(cont_regs)

    # Increased height slightly to accommodate residuals
    figsize = (7.5 * Nregs, 7.5 * Nspec)
    fig = pyplot.figure(figsize=figsize, constrained_layout=True)

    # Create outer grid for spectra (rows) and regions (cols)
    outer_grid = fig.add_gridspec(Nspec, Nregs, hspace=0.1, wspace=0.1)

    best_model_idx = samples['logP'].argmax()

    for i, spectrum in enumerate(spectra.spectra):
        _, wave, _, flux, err = spectrum()
        model_samples = samples[f'{spectrum.name}_model']

        for j in range(Nregs):
            cont_reg = cont_regs[j]

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
                plot_kwargs=plot_kwargs,
                show_ylabel=(j == 0),
                show_rest_labels=(i == 0),
                resid_ax=resid_ax,
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


from copy import deepcopy


def plotRegionSingle(
    config: dict,
    spec,
    region: Tuple[float, float],
    model_args: tuple,
    samples: dict,
    ax: pyplot.Axes = None,
    plot_kwargs: dict | None = None,
    resid_ax: pyplot.Axes = None,
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
    """
    if ax is None:
        fig = pyplot.figure(figsize=(10, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0)
        ax = fig.add_subplot(gs[0])
        resid_ax = fig.add_subplot(gs[1], sharex=ax)

    # Unpack necessary info from model_args
    # spectra, matrices, linetypes_all, line_centers, line_estimates_eq, cont_regs, cont_guesses
    _, _, _, line_centers, _, cont_regs, _ = model_args

    # Get spectrum data
    spec_r = deepcopy(spec)
    spec_r.restrict(cont_regs)
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
        show_rest_labels=True,
        resid_ax=resid_ax,
    )


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
    for group in config['Groups'].items():
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
