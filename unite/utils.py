"""
Utility Functions
"""

# Typing
from typing import List

# Import packages
import copy

# Astronomy packages
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np
import jax.numpy as jnp

# Spectra class
from unite import defaults
from unite.spectra import Spectra, Spectrum


def restrictConfig(
    config: dict, spectra: Spectra, linedet: u.Quantity = defaults.LINEDETECT
) -> List:
    """
    Restrict the configuration to only include lines that are covered by the spectra

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    linedet : u.Quantity, optional
        Padding around the lines necessary to cover the line
        In velocity space

    Returns
    -------
    list
        Updated configuration
    """
    # Parse config['Region'] if it exists
    # if 'Region' in config:
    #     config_region = u.Quantity(config['Region'], config['Unit']).to(spectra.λ_unit).value
    #     config_region = config_region * (1 + spectra.redshift_initial)

    # Set the default linetype as narrow
    for group in config['Groups'].values():
        for species in group['Species']:
            if 'LineType' not in species:
                species['LineType'] = 'narrow'

    # Add additional components
    new_config = copy.deepcopy(config)
    for group in config['Groups'].values():
        for species in group['Species']:
            if 'AdditionalComponents' in species:
                # Iterate over additional components
                for comp, dest in species['AdditionalComponents'].items():
                    # Add it to the correct group
                    new_group = new_config['Groups'][dest]

                    # Check if this component already exists in the destination group
                    # to prevent re-expansion
                    exists = False
                    for s in new_group['Species']:
                        if s['Name'] == species['Name'] and s.get('LineType') == comp:
                            exists = True
                            break

                    if exists:
                        continue

                    # Get copy of species
                    new_species = copy.deepcopy(species)

                    # Remove additional component and add LineType
                    if 'AdditionalComponents' in new_species:
                        new_species.pop('AdditionalComponents')
                    new_species['LineType'] = comp

                    # Add the new species
                    new_group['Species'].append(new_species)

    # Initialize dictionary
    config = copy.deepcopy(new_config)

    # Effective resolution
    lineres = (linedet / consts.c).to(u.dimensionless_unscaled).value

    # Loop over config
    new_groups = {}
    for gname, group in config['Groups'].items():
        new_species = []
        for species in group['Species']:
            new_lines = []
            for line in species['Lines']:
                # Compute line wavelength
                linewav = (line['Wavelength'] * u.Unit(config['Unit'])).to(
                    spectra.λ_unit
                )

                # Redshift the line
                linewav = linewav * (1 + spectra.redshift_initial)
                linewidth = linewav * lineres

                # Compute boundaries
                low, high = (linewav - linewidth).value, (linewav + linewidth).value

                # Check coverage
                if jnp.logical_or.reduce(
                    jnp.array([s.coverage(low, high).any() for s in spectra.spectra])
                ):
                    # dont add lines outside config region
                    # if 'Region' in config:
                    #     if (low < config_region[0]) | (high > config_region[1]):
                    #         continue
                    new_lines.append(line)

            # Add species only if it has remaining lines
            if new_lines:
                species['Lines'] = new_lines
                new_species.append(species)

        # Add group only if it has remaining species
        if new_species:
            group['Species'] = new_species
            new_groups[gname] = group

    # Return the updated config
    return config


# TODO: Maybe start with what we have from GELATO?
def validateConfig(config: dict) -> None:
    """
    Validate configution file is valid.
    Will raise and error if it is not

    Parameters
    ----------
    config : dict
        Configuration of emission lines

    Returns
    -------
    None
    """
    if False:
        raise  # What kind of error?
        exit(1)
    return


import os
from pathlib import Path
from astropy.utils.data import download_file
from astropy.table import Table


def download_spectra(
    spectrum_files: list,
    table_csv: str | None = None,
    #    default_url_prefix: str = "https://zenodo.org/records/15472354/files/",
    default_url_prefix: str = 'https://s3.amazonaws.com/msaexp-nirspec/extractions',
    # Updated 19 Jan 2026
    version: str = 'v4.5',
    spectra_directory: str | None = None,
) -> Table:
    """
    Ensure spectra are present locally (downloading if needed) and optionally
    return a metadata table for the requested files.

    Parameters
    ----------
    spectrum_files : list
        List of spectrum filenames or paths. If bare filenames are supplied
        and ``spectra_directory`` is given, files are placed in that directory.
    table_csv : str or None
        CSV with metadata (local path or URL). If None, the default public
        catalog on Zenodo is used.
    spectra_directory : str or None
        Target directory for downloaded FITS files when ``spectrum_files`` are
        provided as bare names.

    Notes
    -----
    This keeps backward compatibility: callers that only need to download
    files can ignore the new parameters.
    """

    #    FITS_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}"
    FITS_URL = f'{default_url_prefix}/{{root}}/{{file}}'

    # Normalize paths; allow passing bare filenames with a target directory
    normalized_paths = []
    for sf in spectrum_files:
        p = Path(sf)
        if not p.parent or str(p.parent) == '.':
            if spectra_directory is not None:
                p = Path(spectra_directory) / p.name
        normalized_paths.append(p)

    # If everything already exists locally, optionally return table early
    # if all(p.exists() for p in normalized_paths) and not return_table:
    #     return None

    if table_csv is None:
        table_csv = os.path.join(
            default_url_prefix, f'dja_msaexp_emission_lines_{version}.csv.gz'
        )

    p = Path(table_csv)
    if p.exists():
        tab = Table.read(str(p), format='csv')
    else:
        print('Downloading spectra csv table:', table_csv)
        tab = Table.read(download_file(table_csv, cache=True), format='csv')

    # Ensure target directory exists
    Path(normalized_paths[0]).parent.mkdir(parents=True, exist_ok=True)

    selected_rows = []

    for p in normalized_paths:
        fname = p.name
        row_match = tab[tab['file'] == fname]
        if len(row_match) == 0:
            raise ValueError(f'Spectrum file {fname} not found in table {table_csv}')

        if not p.exists():
            url = FITS_URL.format(**row_match[0])
            print(f'Downloading spectrum: {p} from {url}')
            os.rename(download_file(url, cache=False, show_progress=True), p)

        selected_rows.append(row_match[0])

    # shorten grating names
    tab = Table(rows=selected_rows)
    # Each element in tab['grating'] is already a string; split directly
    tab['grating'] = [g.split('_')[0] for g in tab['grating']]

    # Add spectra_directory column
    tab['spectra_directory'] = [str(p.parent) for p in normalized_paths]

    return tab


def masklines(
    config: dict,
    spectra: Spectrum,
    region: np.ndarray,
    broad_mask: u.Quantity = 3000 * u.km / u.s,
    narrow_mask: u.Quantity = 300 * u.km / u.s,
    broad_species: list = ['HI'],
    wave_unit: u.Unit = u.micron,
) -> np.ndarray:
    """
    Create a mask for emission lines

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array
    redshift : float
        Redshift of the source
    config : dict
        Configuration of emission lines
    region : np.ndarray
        Boundary of the continuum region [min, max]
    broad_mask : u.Quantity
        Masking width for broad lines (velocity). Default 5000 km/s.
    narrow_mask : u.Quantity
        Masking width for narrow lines (velocity)
    broad_species : list, optional
        List of species to consider as broad. If None, all 'broad' lines are masked with broad_mask.
        If provided, only 'broad' lines of these species are masked with broad_mask.
    wave_unit : u.Unit, optional
        Unit of the wavelength array, defaults to micron

    Returns
    -------
    np.ndarray
        Boolean mask (True = Continuum, False = Line)
    """
    redshift = spectra.redshift_initial
    wave = spectra.wavelength.to(wave_unit).value

    # Compute redshift factor
    opz = 1 + redshift

    # Convert masks to dimensionless padding
    pad_broad = (broad_mask / consts.c).to(u.dimensionless_unscaled).value
    pad_narrow = (narrow_mask / consts.c).to(u.dimensionless_unscaled).value

    # Extract the region
    low_r, high_r = region
    mask = np.logical_and(low_r < wave, wave < high_r)

    # Mask each line
    λ_unit_config = u.Unit(config['Unit'])

    for group in config['Groups'].values():
        for species in group['Species']:
            # Determine line type
            line_type = species.get('LineType', 'narrow')
            is_broad = line_type == 'broad'

            # Filter by species if list provided
            if is_broad and (broad_species is not None):
                if species['Name'] not in broad_species:
                    is_broad = False

            # Select padding
            pad = pad_broad if is_broad else pad_narrow

            for line in species['Lines']:
                # Compute line wavelength in observed frame
                linewav = (line['Wavelength'] * λ_unit_config).to(wave_unit).value * opz

                # Effective padding width
                width = linewav * pad

                # Mask
                l, h = linewav - width, linewav + width
                linemask = np.logical_and(l < wave, wave < h)
                mask = np.logical_and(mask, np.invert(linemask))

    return mask


def deduplicate_config_lines(config: dict) -> dict:
    """
    Remove duplicate lines from config (fixes dotfit bug)

    Some versions of dotfit generate duplicate lines in the config,
    which causes errors when saving results. This function deduplicates
    lines based on (Wavelength, RelStrength) tuples.

    Parameters
    ----------
    config : dict
        Config dictionary with potential duplicate lines

    Returns
    -------
    dict
        Config with deduplicated lines

    Examples
    --------
    >>> from unite.utils import deduplicate_config_lines
    >>> config_clean = deduplicate_config_lines(config)
    >>> results = NIRSpecFit(config_clean, ...)
    """
    config = copy.deepcopy(config)

    total_before = 0
    total_after = 0

    for group_name, group in config.get('Groups', {}).items():
        for species in group.get('Species', []):
            lines = species.get('Lines', [])
            total_before += len(lines)

            # Deduplicate based on (Wavelength, RelStrength) tuples
            seen = set()
            unique_lines = []

            for line in lines:
                key = (line['Wavelength'], line.get('RelStrength'))
                if key not in seen:
                    seen.add(key)
                    unique_lines.append(line)

            species['Lines'] = unique_lines
            total_after += len(unique_lines)

    if total_before > total_after:
        print(f"Deduplicated config: {total_before} -> {total_after} lines "
              f"({total_before - total_after} duplicates removed)")

    return config
