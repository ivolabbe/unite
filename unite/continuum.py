"""
Continuum models for spectral fitting.

Provides an abstract base class and concrete implementations for different
continuum models (linear, blackbody, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import jax.numpy as jnp
import numpyro.distributions as dist

from unite import priors, optimized, defaults


class ContinuumModel(ABC):
    """Abstract base class for continuum models.

    All continuum models evaluate in REST-FRAME wavelengths.
    model.py handles the redshift conversion.
    """

    @abstractmethod
    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Sample continuum parameters using NumPyro sample function.

        This method handles all parameter sampling internally, including
        any plates for per-region parameters (like LinearContinuum).

        Parameters
        ----------
        sample_fn : callable
            NumPyro sample function (numpyro.sample)
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) in REST-FRAME wavelengths

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary of sampled parameter values
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
        extrapolate: bool = False,
    ) -> jnp.ndarray:
        """
        Evaluate continuum at REST-FRAME wavelengths.

        Parameters
        ----------
        wave_rest : jnp.ndarray
            REST-FRAME wavelength values in microns
        params : Dict[str, jnp.ndarray]
            Dictionary of sampled parameter values (from sample_params)
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) in REST-FRAME wavelengths
        extrapolate : bool
            If True, extrapolate continuum outside fitted regions (default False).
            Only applies to LinearContinuum; other models ignore this parameter.

        Returns
        -------
        jnp.ndarray
            Continuum flux values at wavelengths
        """
        pass


class LinearContinuum(ContinuumModel):
    """Piecewise linear continuum model.

    Evaluates in REST-FRAME wavelengths (like all continuum models).
    """

    def __init__(self, cont_guesses: jnp.ndarray):
        """
        Initialize linear continuum model.

        Parameters
        ----------
        cont_guesses : jnp.ndarray
            Initial guesses for continuum heights
        """
        self.cont_guesses = cont_guesses

    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Sample linear continuum parameters (angle and offset per region).

        Parameters
        ----------
        sample_fn : callable
            NumPyro sample function
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) in REST-FRAME wavelengths

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary with 'cont_angle' (Nc,) and 'cont_offset' (Nc,) arrays
        """
        from numpyro import plate

        Nc = len(cont_regs)
        with plate(f'Nc = {Nc}', Nc):
            angles = sample_fn('cont_angle', priors.angle_prior())
            offsets = sample_fn('cont_offset', priors.height_prior(self.cont_guesses))

        return {'cont_angle': angles, 'cont_offset': offsets}

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
        extrapolate: bool = False,
    ) -> jnp.ndarray:
        """
        Evaluate piecewise linear continuum in REST-FRAME.

        Parameters
        ----------
        wave_rest : jnp.ndarray
            REST-FRAME wavelength values in microns
        params : Dict[str, jnp.ndarray]
            Dictionary with 'cont_angle' and 'cont_offset' arrays
        cont_regs : jnp.ndarray
            Continuum regions (Nc, 2) in REST-FRAME wavelengths
        extrapolate : bool
            If True, extrapolate linear segments outside fitted regions (default False)

        Returns
        -------
        jnp.ndarray
            Sum of all linear continuum segments
        """
        cont_centers = cont_regs.mean(axis=1)
        return optimized.linearContinua(
            wave_rest,
            cont_centers,
            params['cont_angle'],
            params['cont_offset'],
            cont_regs,
            extrapolate=extrapolate,
        ).sum(1)


class BlackbodyContinuum(ContinuumModel):
    """Pure blackbody continuum (beta=0).

    Evaluates in REST-FRAME wavelengths (like all continuum models).
    """

    def __init__(
        self,
        amplitude_guess: float,
        pivot_micron: float = 1.0,
        temp_type: str = 'default',
        temperature: tuple = None,
        temperature_guess: float = None,
        name: str = '',
    ):
        """
        Initialize blackbody continuum model.

        Parameters
        ----------
        amplitude_guess : float
            Initial guess for amplitude
        pivot_micron : float
            Pivot wavelength for normalization (microns)
        temp_type : str
            Temperature range type ('hot', 'warm', 'default'). Ignored if temperature is provided.
        temperature : tuple, optional
            Direct specification of temperature bounds (low, high) in Kelvin.
            If provided, overrides temp_type.
        temperature_guess : float, optional
            Initial guess for temperature (Kelvin). Used to create more informative prior.
        name : str
            Name prefix for parameters (e.g., 'bb1' for composite continuum)
        """
        self.amplitude_guess = amplitude_guess
        self.pivot_micron = pivot_micron
        self.name = name if name else 'bb'
        self.temperature_guess = temperature_guess

        # Store temperature bounds - either from direct specification or from temp_type
        if temperature is not None:
            self.temp_bounds = temperature
        else:
            self.temp_bounds = defaults.temperature[temp_type]

    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Sample blackbody parameters (amplitude and temperature).

        Parameters
        ----------
        sample_fn : callable
            NumPyro sample function
        cont_regs : jnp.ndarray
            Continuum regions (unused for blackbody)

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary with '{name}_amplitude' and '{name}_temperature' scalars
        """
        amplitude = sample_fn(f'{self.name}_amplitude', priors.amplitude_prior(self.amplitude_guess))

        # Use uniform prior for temperature (initialization will be handled separately via init_strategy)
        temperature = sample_fn(
            f'{self.name}_temperature', dist.Uniform(low=self.temp_bounds[0], high=self.temp_bounds[1])
        )
        return {f'{self.name}_amplitude': amplitude, f'{self.name}_temperature': temperature}

    def get_init_values(self) -> Dict[str, float]:
        """
        Get initial values for MCMC sampling.

        Returns
        -------
        Dict[str, float]
            Dictionary with '{name}_amplitude' and '{name}_temperature' initial values
        """
        init_vals = {f'{self.name}_amplitude': self.amplitude_guess}
        if self.temperature_guess is not None:
            init_vals[f'{self.name}_temperature'] = self.temperature_guess
        else:
            # Use midpoint of temperature range if no guess provided
            init_vals[f'{self.name}_temperature'] = (self.temp_bounds[0] + self.temp_bounds[1]) / 2.0
        return init_vals

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
        extrapolate: bool = False,
    ) -> jnp.ndarray:
        """
        Evaluate blackbody at REST-FRAME wavelengths.

        Parameters
        ----------
        wave_rest : jnp.ndarray
            REST-FRAME wavelength values in microns
        params : Dict[str, jnp.ndarray]
            Dictionary with '{name}_amplitude' and '{name}_temperature' parameters
        cont_regs : jnp.ndarray
            Continuum regions (unused for blackbody)
        extrapolate : bool
            Ignored for blackbody (always evaluates everywhere)

        Returns
        -------
        jnp.ndarray
            Blackbody flux values at wavelengths
        """
        return params[f'{self.name}_amplitude'] * optimized.planck_function(
            wave_rest, params[f'{self.name}_temperature'], self.pivot_micron
        )


class ModifiedBlackbodyContinuum(ContinuumModel):
    """Modified blackbody with emissivity index (dust-like).

    Evaluates in REST-FRAME wavelengths (like all continuum models).
    """

    def __init__(
        self,
        amplitude_guess: float,
        pivot_micron: float = 1.0,
        temp_type: str = 'default',
        temperature: tuple = None,
        beta: tuple = None,
        temperature_guess: float = None,
        name: str = '',
    ):
        """
        Initialize modified blackbody continuum model.

        Parameters
        ----------
        amplitude_guess : float
            Initial guess for amplitude
        pivot_micron : float
            Pivot wavelength for normalization (microns)
        temp_type : str
            Temperature range type ('hot', 'warm', 'default'). Ignored if temperature is provided.
        temperature : tuple, optional
            Direct specification of temperature bounds (low, high) in Kelvin.
            If provided, overrides temp_type.
        beta : tuple, optional
            Direct specification of beta (emissivity index) bounds (low, high).
            If not provided, uses defaults.beta['default'].
        temperature_guess : float, optional
            Initial guess for temperature (Kelvin). Used to create more informative prior.
        name : str
            Name prefix for parameters (e.g., 'mbb1' for composite continuum)
        """
        self.amplitude_guess = amplitude_guess
        self.pivot_micron = pivot_micron
        self.name = name if name else 'mbb'
        self.temperature_guess = temperature_guess

        # Store temperature bounds
        if temperature is not None:
            self.temp_bounds = temperature
        else:
            self.temp_bounds = defaults.temperature[temp_type]

        # Store beta bounds
        if beta is not None:
            self.beta_bounds = beta
        else:
            self.beta_bounds = defaults.beta['default']

    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Sample modified blackbody parameters (amplitude, temperature, beta).

        Parameters
        ----------
        sample_fn : callable
            NumPyro sample function
        cont_regs : jnp.ndarray
            Continuum regions (unused for modified blackbody)

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary with '{name}_amplitude', '{name}_temperature', '{name}_beta' scalars
        """
        amplitude = sample_fn(f'{self.name}_amplitude', priors.amplitude_prior(self.amplitude_guess))
        temperature = sample_fn(
            f'{self.name}_temperature', dist.Uniform(low=self.temp_bounds[0], high=self.temp_bounds[1])
        )
        beta = sample_fn(f'{self.name}_beta', dist.Uniform(low=self.beta_bounds[0], high=self.beta_bounds[1]))
        return {
            f'{self.name}_amplitude': amplitude,
            f'{self.name}_temperature': temperature,
            f'{self.name}_beta': beta,
        }

    def get_init_values(self) -> Dict[str, float]:
        """Get initial values for MCMC sampling."""
        init_vals = {
            f'{self.name}_amplitude': self.amplitude_guess,
            f'{self.name}_beta': (self.beta_bounds[0] + self.beta_bounds[1]) / 2.0,
        }
        if self.temperature_guess is not None:
            init_vals[f'{self.name}_temperature'] = self.temperature_guess
        else:
            init_vals[f'{self.name}_temperature'] = (self.temp_bounds[0] + self.temp_bounds[1]) / 2.0
        return init_vals

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
        extrapolate: bool = False,
    ) -> jnp.ndarray:
        """
        Evaluate modified blackbody at REST-FRAME wavelengths.

        Parameters
        ----------
        wave_rest : jnp.ndarray
            REST-FRAME wavelength values in microns
        params : Dict[str, jnp.ndarray]
            Dictionary with 'mbb_amplitude', 'mbb_temperature', 'mbb_beta' parameters
        cont_regs : jnp.ndarray
            Continuum regions (unused for modified blackbody)
        extrapolate : bool
            Ignored for modified blackbody (always evaluates everywhere)

        Returns
        -------
        jnp.ndarray
            Modified blackbody flux values at wavelengths
        """
        return params[f'{self.name}_amplitude'] * optimized.modified_blackbody(
            wave_rest, params[f'{self.name}_temperature'], params[f'{self.name}_beta'], self.pivot_micron
        )


class AttenuatedBlackbodyContinuum(ContinuumModel):
    """
    Attenuated Planck blackbody with power-law dust extinction.

    Models: F_λ = A × B_λ(T) × exp(-τ_V × (λ/0.55μm)^α)
    where α is a free parameter (typically -0.4 to -1.6).
    """

    def __init__(
        self,
        amplitude_guess: float,
        pivot_micron: float = 1.0,
        temp_type: str = 'default',
        tau_type: str = 'default',
        alpha_type: str = 'default',
        temperature: tuple = None,
        tau_v: tuple = None,
        alpha: tuple = None,
        temperature_guess: float = None,
        name: str = '',
    ):
        """
        Initialize attenuated blackbody continuum model.

        Parameters
        ----------
        amplitude_guess : float
            Initial guess for amplitude
        pivot_micron : float
            Pivot wavelength for normalization (microns)
        temp_type : str
            Temperature prior type: 'hot', 'warm', 'dust', or 'default'. Ignored if temperature is provided.
        tau_type : str
            Optical depth prior type: 'low', 'moderate', 'high', or 'default'. Ignored if tau_v is provided.
        alpha_type : str
            Attenuation slope prior type: 'mw', 'lmc', 'smc', or 'default'. Ignored if alpha is provided.
        temperature : tuple, optional
            Direct specification of temperature bounds (low, high) in Kelvin.
            If provided, overrides temp_type.
        tau_v : tuple, optional
            Direct specification of optical depth bounds (low, high).
            If provided, overrides tau_type.
        alpha : tuple, optional
            Direct specification of attenuation slope bounds (low, high).
            If provided, overrides alpha_type.
        temperature_guess : float, optional
            Initial guess for temperature (Kelvin). Used to create more informative prior.
        name : str
            Identifier prefix for parameter names (default: 'abb')
        """
        self.amplitude_guess = amplitude_guess
        self.pivot_micron = pivot_micron
        self.name = name if name else 'abb'
        self.temperature_guess = temperature_guess

        # Store temperature bounds
        if temperature is not None:
            self.temp_bounds = temperature
        else:
            self.temp_bounds = defaults.temperature[temp_type]

        # Store tau_v bounds
        if tau_v is not None:
            self.tau_v_bounds = tau_v
        else:
            self.tau_v_bounds = defaults.tau_v[tau_type]

        # Store alpha bounds
        if alpha is not None:
            self.alpha_bounds = alpha
        else:
            self.alpha_bounds = defaults.alpha_atten[alpha_type]

    def sample_params(self, sample_fn, cont_regs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Sample attenuated blackbody parameters (amplitude, temperature, tau_v, alpha).

        Parameters
        ----------
        sample_fn : callable
            NumPyro sample function
        cont_regs : jnp.ndarray
            Continuum regions (unused for attenuated blackbody)

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary with '{name}_amplitude', '{name}_temperature', '{name}_tau_v', '{name}_alpha' scalars
        """
        amplitude = sample_fn(f'{self.name}_amplitude', priors.amplitude_prior(self.amplitude_guess))
        temperature = sample_fn(
            f'{self.name}_temperature', dist.Uniform(low=self.temp_bounds[0], high=self.temp_bounds[1])
        )
        tau_v = sample_fn(f'{self.name}_tau_v', dist.Uniform(low=self.tau_v_bounds[0], high=self.tau_v_bounds[1]))
        alpha = sample_fn(f'{self.name}_alpha', dist.Uniform(low=self.alpha_bounds[0], high=self.alpha_bounds[1]))
        return {
            f'{self.name}_amplitude': amplitude,
            f'{self.name}_temperature': temperature,
            f'{self.name}_tau_v': tau_v,
            f'{self.name}_alpha': alpha,
        }

    def get_init_values(self) -> Dict[str, float]:
        """Get initial values for MCMC sampling."""
        init_vals = {
            f'{self.name}_amplitude': self.amplitude_guess,
            f'{self.name}_tau_v': (self.tau_v_bounds[0] + self.tau_v_bounds[1]) / 2.0,
            f'{self.name}_alpha': (self.alpha_bounds[0] + self.alpha_bounds[1]) / 2.0,
        }
        if self.temperature_guess is not None:
            init_vals[f'{self.name}_temperature'] = self.temperature_guess
        else:
            init_vals[f'{self.name}_temperature'] = (self.temp_bounds[0] + self.temp_bounds[1]) / 2.0
        return init_vals

    def evaluate(
        self,
        wave_rest: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        cont_regs: jnp.ndarray,
        extrapolate: bool = False,
    ) -> jnp.ndarray:
        """
        Evaluate attenuated blackbody at REST-FRAME wavelengths.

        Parameters
        ----------
        wave_rest : jnp.ndarray
            REST-FRAME wavelength values in microns
        params : Dict[str, jnp.ndarray]
            Dictionary with 'abb_amplitude', 'abb_temperature', 'abb_tau_v', 'abb_alpha' parameters
        cont_regs : jnp.ndarray
            Continuum regions (unused for attenuated blackbody)
        extrapolate : bool
            Ignored for attenuated blackbody (always evaluates everywhere)

        Returns
        -------
        jnp.ndarray
            Attenuated blackbody flux values at wavelengths
        """
        return params[f'{self.name}_amplitude'] * optimized.attenuated_planck(
            wave_rest,
            params[f'{self.name}_temperature'],
            params[f'{self.name}_tau_v'],
            params[f'{self.name}_alpha'],
            self.pivot_micron,
        )


def parse_continuum_config(
    config: dict, cont_guesses: jnp.ndarray, spectra=None, cont_regs=None
) -> List[ContinuumModel]:
    """
    Parse continuum config and return list of continuum models.

    Config format (single model with predefined types):
    {
        "continuum": {
            "type": "blackbody",
            "pivot_micron": 0.5,  # optional, default 0.5
            "temp_type": "hot"  # optional: 'hot', 'warm', 'dust', 'default'
        }
    }

    Config format (single model with direct bounds):
    {
        "continuum": {
            "type": "blackbody",
            "pivot_micron": 0.5,
            "temperature": (4000.0, 7000.0)  # Direct (low, high) bounds in Kelvin
        }
    }

    Config format (attenuated blackbody with all parameters):
    {
        "continuum": {
            "type": "attenuated_blackbody",
            "temperature": (5000.0, 10000.0),  # Optional direct bounds
            "tau_v": (0.0, 2.0),  # Optional direct bounds
            "alpha": (-1.2, -0.6)  # Optional direct bounds
        }
    }

    Config format (composite continuum - multiple models):
    {
        "continuum": [
            {"type": "linear"},
            {"type": "blackbody", "temperature": (4000, 7000)}
        ]
    }

    If no continuum key, defaults to [LinearContinuum].

    Note: Direct parameter bounds (e.g., temperature, tau_v, alpha, beta) override
    the corresponding type parameters (e.g., temp_type, tau_type, alpha_type).

    Parameters
    ----------
    config : dict
        UNITE configuration dictionary
    cont_guesses : jnp.ndarray
        Initial guesses for continuum heights from spectrum

    Returns
    -------
    List[ContinuumModel]
        List of continuum model instances (usually single element)
    """
    if 'continuum' not in config:
        return [LinearContinuum(cont_guesses)]

    # Estimate temperature from spectrum if available
    temperature_guess = None
    if spectra is not None and cont_regs is not None:
        try:
            from unite.initial import estimateBlackbodyTemperature
            temperature_guess = estimateBlackbodyTemperature(spectra, cont_guesses, cont_regs)
        except ImportError:
            # Function not implemented yet, use default
            pass

    cont_cfg = config['continuum']

    # Check if it's a list of models (composite)
    if isinstance(cont_cfg, list):
        models = []
        for i, model_cfg in enumerate(cont_cfg):
            # Add unique name suffix for composite models
            model = _parse_single_continuum(model_cfg, cont_guesses, temperature_guess, name_suffix=str(i + 1))
            models.append(model)
        return models
    else:
        # Single model
        return [_parse_single_continuum(cont_cfg, cont_guesses, temperature_guess)]


def _parse_single_continuum(
    cont_cfg: dict, cont_guesses: jnp.ndarray, temperature_guess: float = None, name_suffix: str = ''
) -> ContinuumModel:
    """Parse a single continuum model configuration.

    Supports both predefined types (e.g., temp_type='hot') and direct bound specification
    (e.g., temperature=(4000, 7000)).
    """
    cont_type = cont_cfg.get('type', 'linear').lower()
    pivot = cont_cfg.get('pivot_micron', 0.5)

    # For BB/MBB, use cont_guesses mean as amplitude guess
    amp_guess = float(cont_guesses.mean()) if len(cont_guesses) > 0 else 1.0

    if cont_type == 'linear':
        return LinearContinuum(cont_guesses)
    elif cont_type == 'blackbody':
        name = f'bb{name_suffix}' if name_suffix else 'bb'
        # Support both temp_type string and direct temperature tuple
        temp_type = cont_cfg.get('temp_type', 'default')
        temperature = cont_cfg.get('temperature', None)
        return BlackbodyContinuum(
            amp_guess,
            pivot,
            temp_type=temp_type,
            temperature=temperature,
            temperature_guess=temperature_guess,
            name=name,
        )
    elif cont_type == 'modified_blackbody':
        name = f'mbb{name_suffix}' if name_suffix else 'mbb'
        # Support both type strings and direct tuples
        temp_type = cont_cfg.get('temp_type', 'default')
        temperature = cont_cfg.get('temperature', None)
        beta = cont_cfg.get('beta', None)
        return ModifiedBlackbodyContinuum(
            amp_guess,
            pivot,
            temp_type=temp_type,
            temperature=temperature,
            beta=beta,
            temperature_guess=temperature_guess,
            name=name,
        )
    elif cont_type == 'attenuated_blackbody':
        name = f'abb{name_suffix}' if name_suffix else 'abb'
        # Support both type strings and direct tuples
        temp_type = cont_cfg.get('temp_type', 'default')
        tau_type = cont_cfg.get('tau_type', 'default')
        alpha_type = cont_cfg.get('alpha_type', 'default')
        temperature = cont_cfg.get('temperature', None)
        tau_v = cont_cfg.get('tau_v', None)
        alpha = cont_cfg.get('alpha', None)
        return AttenuatedBlackbodyContinuum(
            amp_guess,
            pivot,
            temp_type=temp_type,
            tau_type=tau_type,
            alpha_type=alpha_type,
            temperature=temperature,
            tau_v=tau_v,
            alpha=alpha,
            temperature_guess=temperature_guess,
            name=name,
        )
    else:
        raise ValueError(f'Unknown continuum type: {cont_type}')


def compute_continuum_regions(config: dict, spectra) -> jnp.ndarray:
    """
    Compute continuum regions from manual region specification in config.

    Handles include/exclude logic and rest-frame/observed-frame conversion.

    Parameters
    ----------
    config : dict
        Config with continuum.regions specification
    spectra : NIRSpecSpectra
        Spectra object

    Returns
    -------
    jnp.ndarray
        Continuum regions in observed frame (microns), shape (N, 2)
    """
    from astropy import units as u
    import numpy as np

    regions_spec = config['continuum']['regions']
    include = regions_spec.get('include')
    exclude = regions_spec.get('exclude', [])
    rest_frame = regions_spec.get('rest_frame', True)

    # Get unit from config (default Angstrom)
    unit = u.Unit(config.get('Unit', 'Angstrom'))

    # Normalize include to list of tuples
    if isinstance(include, tuple) and len(include) == 2 and isinstance(include[0], (int, float)):
        include = [include]

    # Normalize exclude to list of tuples
    if isinstance(exclude, tuple) and len(exclude) == 2:
        exclude = [exclude]

    # Convert to observed frame microns
    opz = 1 + spectra.redshift_initial

    def to_obs_micron(wavelength_val):
        """Convert wavelength value to observed-frame microns"""
        wave = wavelength_val * unit  # Apply unit from config
        if rest_frame:
            wave = wave * opz  # Convert rest to observed
        return wave.to(u.micron).value

    # Process include regions
    include_regions = []
    for inc in include:
        low, high = to_obs_micron(inc[0]), to_obs_micron(inc[1])
        include_regions.append([low, high])

    # Process exclude regions
    exclude_regions = []
    for exc in exclude:
        low, high = to_obs_micron(exc[0]), to_obs_micron(exc[1])
        exclude_regions.append([low, high])

    # Apply exclusions to inclusions
    final_regions = []
    for inc_low, inc_high in include_regions:
        # Start with the full include region
        current_regions = [[inc_low, inc_high]]

        # Subtract each exclude region
        for exc_low, exc_high in exclude_regions:
            new_regions = []
            for reg_low, reg_high in current_regions:
                # Check if exclude overlaps with current region
                if exc_high <= reg_low or exc_low >= reg_high:
                    # No overlap - keep region as is
                    new_regions.append([reg_low, reg_high])
                elif exc_low <= reg_low and exc_high >= reg_high:
                    # Exclude completely covers region - remove it
                    pass
                elif exc_low > reg_low and exc_high < reg_high:
                    # Exclude is inside region - split into two
                    new_regions.append([reg_low, exc_low])
                    new_regions.append([exc_high, reg_high])
                elif exc_low <= reg_low:
                    # Exclude overlaps left side
                    new_regions.append([exc_high, reg_high])
                else:
                    # Exclude overlaps right side
                    new_regions.append([reg_low, exc_low])
            current_regions = new_regions

        final_regions.extend(current_regions)

    # Convert to jax array
    if not final_regions:
        raise ValueError("No continuum regions remain after applying include/exclude logic")

    return jnp.array(final_regions)
