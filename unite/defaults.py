"""
Default Values
"""

# Typing
from enum import Enum
from typing import Final

# Astropy packages
from astropy import units as u

# JAX packages
import jax.numpy as jnp


# Fitting mode: which pixels enter the likelihood
class FittingMode(str, Enum):
    """How the spectrum is restricted for fitting."""

    LINES = 'lines'  # Fit around emission line regions (default for linear continuum)
    FULL = 'full'  # Fit entire spectral range (default for physical continua)
    CONTINUUM = 'continuum'  # Fit everywhere except masked emission lines


# Constants used for line detection, line padding, and continuum regions
LINEDETECT: u.Quantity = 1_000 * (u.km / u.s)
LINEPAD: u.Quantity = 5_000 * (u.km / u.s)
CONTINUUM: u.Quantity = 10_000 * (u.km / u.s)

# Dictionary that defines mapping from integers to line types
linetypes: list = ['narrow', 'broad', 'lorentzian', 'exponential', 'absorption', 'emission', 'outflow']
LINETYPES: Final[dict] = {line: i for i, line in enumerate(linetypes)}

# Define the Flux priors (scale relative to the guess)
flux: dict[str, tuple[float]] = {
    'narrow': (-2, 2),
    'broad': (0, 3),
    'emission': (0, 2),
    'outflow': (-2, 2),
    'absorption': (-2, 0),
    'lorentzian': (0, 3),
    'exponential': (0, 3),
}


# Define the Redshift priors in dimensionless units
δz: Final[float] = 0.005
redshift: dict[str, tuple[float]] = {
    'narrow': (-δz, δz),
    'broad': (-2 * δz, 2 * δz),
    'emission': (-δz, δz),
    'outflow': (-δz, δz),
    'absorption': (-3 * δz, 3 * δz),
    'lorentzian': (-2 * δz, 2 * δz),
    'exponential': (-2 * δz, 2 * δz),
}

# Define the Dispersion priors in km/s
fwhm: dict[str, tuple[float]] = {
    'narrow': (0, 750),
    'broad': (1000, 7500),
    'emission': (0, 750),
    'absorption': (0, 750),
    'outflow': (200, 2500),
    'lorentzian': (200, 7500),
    'exponential': (200, 7500),
}

# Temperature bounds (Kelvin) for blackbody continuum
temperature: dict[str, tuple[float]] = {
    'hot': (15_000.0, 50_000.0),  # Hot stars, AGN
    'warm': (2_000.0, 10_000.0),  # Warm dust
    'cold': (20.0, 1_500.0),  # Cold dust
    'default': (1_000.0, 30_000.0),  # Full range
}


def apply_config_defaults(config: dict) -> None:
    """Apply config-specified default overrides for reproducibility.

    Parses ``config['defaults']`` and sets module-level attributes.
    String values are interpreted as `~astropy.units.Quantity`.

    Examples
    --------
    >>> config['defaults'] = {'CONTINUUM': '10000 km/s'}
    >>> config['defaults'] = {'LINEPAD': '5000 km/s', 'CONTINUUM': '15000 km/s'}
    """
    module_globals = globals()
    for key, val in config.get('defaults', {}).items():
        if key not in module_globals:
            raise ValueError(f'Unknown default: {key!r}')
        if isinstance(val, str):
            val = u.Quantity(val)
        module_globals[key] = val


def convertToArray(priorDict: dict[str, tuple[float]]) -> jnp.ndarray:
    """
    Convert dictionary of priors to JAX Array

    Parameters
    ----------
    priorDict : dict[str, tuple[float]]
        Dictionary of prior values

    Returns
    -------
    jnp.ndarray
        Array of prior values
    """

    # Initialize the array
    out = jnp.zeros((len(set(priorDict)), 2))

    # Fill the array
    for key, val in priorDict.items():
        out = out.at[LINETYPES[key]].set(val)

    return out
