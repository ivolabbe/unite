"""
Default Values
"""

# Typing
from typing import Final

# Astropy packages
from astropy import units as u

# JAX packages
import jax.numpy as jnp

# Constants used for line detection, line padding, and continuum regions
LINEDETECT: u.Quantity = 1_000 * (u.km / u.s)
LINEPAD: u.Quantity = 3_500 * (u.km / u.s)
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
    'broad': (1000, 5500),
    'emission': (0, 750),
    'absorption': (0, 750),
    'outflow': (200, 2500),
    'lorentzian': (200, 5500),
    'exponential': (200, 5500),
}

# Temperature bounds (Kelvin) for blackbody continuum
temperature: dict[str, tuple[float]] = {
    'hot': (15_000.0, 50_000.0),  # Hot stars, AGN
    'warm': (2_000.0, 10_000.0),  # Warm dust
    'cold': (20.0, 1500.0),  # dust
    'default': (1_000.0, 30_000.0),
}

# Emissivity index (beta) bounds for modified blackbody
beta: dict[str, tuple[float]] = {'default': (-2.0, 2.0)}

# V-band optical depth (tau_v) bounds for attenuated blackbody
tau_v: dict[str, tuple[float]] = {
    'low': (0.0, 1.0),  # Low extinction
    'moderate': (0.0, 3.0),  # Moderate extinction
    'high': (0.0, 10.0),  # High extinction (heavily obscured)
    'default': (0.0, 3.0),
}

# Attenuation power-law slope (alpha) for attenuated blackbody
# α ~ -0.4 (Milky Way), -0.7 (LMC), -1.6 (SMC), steeper than -2.0 (very steep)
alpha_atten: dict[str, tuple[float]] = {
    'mw': (-0.6, -0.2),  # Milky Way-like (shallow slope)
    'lmc': (-0.9, -0.5),  # LMC-like (intermediate)
    'smc': (-1.8, -1.4),  # SMC-like (steep slope)
    'default': (-2.0, -0.4),  # Full range (MW to very steep)
}

# Default lines to mask in continuum-only mode
# Format: (name, wavelength_angstrom, line_type)
# line_type determines padding: 'broad' uses LINEPAD, 'narrow' uses LINEDETECT
DEFAULT_MASK_LINES: list = [
    # Balmer (broad)
    ('Ha', 6564.6, 'broad'),
    ('Hb', 4862.7, 'broad'),
    ('Hg', 4341.7, 'broad'),
    # Balmer (narrow)
    ('Hd', 4102.9, 'narrow'),
    ('He', 3971.2, 'narrow'),
    # [OII]
    ('[OII]_3727', 3727.0, 'narrow'),
    ('[OII]_3729', 3729.0, 'narrow'),
    # [OIII]
    ('[OIII]_4959', 4960.0, 'narrow'),
    ('[OIII]_5007', 5008.0, 'narrow'),
    # OI
    ('OI_6302', 6302.0, 'narrow'),
    ('OI_6365', 6365.0, 'narrow'),
    # [SII]
    ('[SII]_6718', 6718.0, 'narrow'),
    ('[SII]_6732', 6732.0, 'narrow'),
    # [SIII]
    ('[SIII]_9071', 9071.0, 'narrow'),
    ('[SIII]_9533', 9533.0, 'narrow'),
    # Paschen series (broad)
    ('Paa', 18756.0, 'broad'),
    ('Pab', 12822.0, 'broad'),
    ('Pag', 10941.0, 'broad'),
    ('Pad', 10052.0, 'broad'),
    # NIR HeI
    ('HeI_10833', 10833.0, 'narrow'),
    ('HeI_20587', 20587.0, 'narrow'),
]


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
