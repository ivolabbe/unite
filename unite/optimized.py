"""
Optimized line profile integrals
"""

# Typing
from typing import Final

# JAX packages
from jax.scipy.special import erf, erfc
from jax import config, jit, vmap, lax, numpy as jnp
import jax

# Conversion factor from FWHM to sigma for variance = 1/2
# σ = fwhm / ( 2 * sqrt( 2 * ln(2) ) )
# σ_halfvar = sqrt(2) * σ
_HALFVAR_SIGMA_TO_FWHM: Final[float] = 2 * jnp.sqrt(jnp.log(2))

# Profile Types
# Profile Types
GAUSSIAN: Final[int] = 0
LORENTZIAN: Final[int] = 1
EXPONENTIAL: Final[int] = 2


@jit
def integrateGaussian(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Gaussian emission lines over wavelength bins.

    Computes the integral of N Gaussian profiles over wavelength bins using
    the error function (Gaussian CDF).

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm : jnp.ndarray
        Gaussian FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Convert FWHM to inverse sigma (variance = 1/2)
    inv_sigma = _HALFVAR_SIGMA_TO_FWHM / fwhm

    # Normalized distance from center
    t_low = (low - center) * inv_sigma
    t_high = (high - center) * inv_sigma

    # Integrate using Gaussian CDF (error function)
    return (erf(t_high) - erf(t_low)) / 2


@jit
def integrateCauchy(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Cauchy (Lorentzian) emission lines over wavelength bins.

    Computes the integral of N Cauchy/Lorentzian profiles over wavelength bins
    using the arctan function (Cauchy CDF).

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm : jnp.ndarray
        Lorentzian FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Convert FWHM to inverse half-width
    inv_hwhm = 2 / fwhm

    # Normalized distance from center
    t_low = (low - center) * inv_hwhm
    t_high = (high - center) * inv_hwhm

    # Integrate using Cauchy CDF (arctan)
    return (jnp.arctan(t_high) - jnp.arctan(t_low)) / jnp.pi


# Pseudo-Voigt profile magic numbers from Thompson+ (1987) DOI:10.1107/S0021889887087090
_VOIGT_FWHM_CS: Final[jnp.ndarray] = jnp.array([1, 2.69268, 2.42843, 4.47163, 0.07842, 1])
_VOIGT_ETA_CS: Final[jnp.ndarray] = jnp.array([1.33603, -0.47719, 0.11116])


@jit
def integrateVoigt(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm_g: jnp.ndarray, fwhm_γ: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Voigt emission lines over wavelength bins.

    Computes the integral of N Voigt profiles (Gaussian convolved with Lorentzian)
    using the pseudo-Voigt approximation from Thompson+ (1987).
    DOI:10.1107/S0021889887087090

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm_g : jnp.ndarray
        Gaussian component FWHM at each line
    fwhm_γ : jnp.ndarray
        Lorentzian component FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Compute effective FWHM for pseudo-Voigt
    pows = jnp.arange(_VOIGT_FWHM_CS.size)
    fwhm = jnp.sum(_VOIGT_FWHM_CS * (fwhm_g**pows) * (fwhm_γ ** pows[::-1])) ** (1 / 5)

    # Compute Lorentzian mixing parameter η
    fwhm_ratio = fwhm_γ / fwhm
    η = jnp.sum(_VOIGT_ETA_CS * (fwhm_ratio ** jnp.arange(1, len(_VOIGT_ETA_CS) + 1)))

    # Weighted sum of Lorentzian and Gaussian components
    lorentzian = η * integrateCauchy(low, high, center, fwhm)
    gaussian = (1 - η) * integrateGaussian(low, high, center, fwhm)

    return lorentzian + gaussian


# Conversion factor from exponential (Laplace) scale to FWHM
# pdf = (1/(2*b)) * exp(-|x - μ|/b)
# max(pdf) = 1/(2*b), half max = 1/(4*b)
# 1/(4*b) = 1/(2*b) * exp(-|x - μ|/b) => exp(-|x - μ|/b) = 1/2
# => |x - μ|/b = ln(2) => FWHM = 2*b*ln(2)
_EXP_SCALE_TO_FWHM: Final[float] = 2 * jnp.log(2)


@jit
def integrateLaplace(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Laplace (double exponential) emission lines over wavelength bins.

    Computes the integral of N Laplace/double exponential profiles over wavelength
    bins using the exponential CDF.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm : jnp.ndarray
        Laplace FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Convert FWHM to scale parameter
    λ = _EXP_SCALE_TO_FWHM / fwhm

    # Normalized distance from center
    t_low = (low - center) * λ
    t_high = (high - center) * λ

    # Laplace CDF: F(t) = 1/2 + 1/2 * sign(t) * (1 - exp(-|t|))
    # Rewritten without sign() for differentiability:
    #   For t >= 0: F(t) = 1 - 1/2 * exp(-t)
    #   For t < 0:  F(t) = 1/2 * exp(t)
    @jit
    def laplace_cdf(t):
        return jnp.where(t >= 0, 1 - 0.5 * jnp.exp(-t), 0.5 * jnp.exp(t))

    return laplace_cdf(t_high) - laplace_cdf(t_low)


# Threshold for when exp(x*x) overflows
_OVERFLOW_THRESHOLD: Final[float] = 26.0 if config.jax_enable_x64 else 9.0


@jit
def _integrandGL(t: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """
    Compute exponential correction integrand for Gaussian-Laplace convolution.

    Evaluates exp(-a²) * [exp(2ta) * erfcx(a+t) - exp(-2ta) * erfcx(a-t)].
    Exploits odd symmetry to cover overflow at large a + t.

    Parameters
    ----------
    t : jnp.ndarray
        Normalized distance from center
    a : jnp.ndarray
        Convolution parameter (σλ/2)

    Returns
    -------
    jnp.ndarray
        Integrand value with numerical stability
    """
    # Exploit odd symmetry: I(-t,a) = -I(t,a)
    # Then we only need one overflow protection
    t_abs = jnp.abs(t)

    # Shorthand terms
    ta = t_abs + a
    twota = 2 * t_abs * a

    # Overflow protection
    posterm = jnp.where(ta > _OVERFLOW_THRESHOLD, 0, jnp.exp(twota) * erfc(ta))

    # Compute for positive t, then use odd symmetry for negative t
    # Replace sign(t) with jnp.where for differentiability
    result_abs = jnp.exp(a * a) * (posterm - jnp.exp(-twota) * erfc(-t_abs + a))
    return jnp.where(t >= 0, result_abs, -result_abs)


@jit
def integrateGaussianLaplace(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm_g: jnp.ndarray, fwhm_l: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate exponentially modified Gaussian (EMG) emission lines over wavelength bins.

    Computes the integral of Gaussian convolved with symmetric double exponential
    (Laplace distribution). This profile combines instrumental broadening (Gaussian)
    with natural broadening (Laplacian). Uses asymptotic approximations for
    numerical stability.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm_g : jnp.ndarray
        Gaussian component FWHM (instrumental broadening)
    fwhm_l : jnp.ndarray
        Laplacian component FWHM (natural broadening)

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # This function breaks for pure Laplace, does that matter? lsf is never zero?

    # Convert FWHM to distribution parameters
    σ = fwhm_g / _HALFVAR_SIGMA_TO_FWHM
    λ = _EXP_SCALE_TO_FWHM / fwhm_l

    # Normalized distance from center
    scale = 1 / σ
    t_low = (low - center) * scale
    t_high = (high - center) * scale

    # Convolution parameter
    a = σ * λ / 2

    # Gaussian component (via error function CDF)
    gaussian_cdf = (erf(t_high) - erf(t_low)) / 2

    # Exponential correction
    exp_correction = _integrandGL(t_high, a) - _integrandGL(t_low, a)

    # Guard against large a, the Gaussian limit
    return gaussian_cdf + jnp.where(a > _OVERFLOW_THRESHOLD, 0, exp_correction / 4)


# Wrappers for lax.switch
def _wrap_gaussian(low, high, center, lsf, fwhm):
    return integrateGaussian(low, high, center, jnp.sqrt(lsf**2 + fwhm**2))


def _wrap_voigt(low, high, center, lsf, fwhm):
    return integrateVoigt(low, high, center, lsf, fwhm)


def _wrap_gaussian_laplace(low, high, center, lsf, fwhm):
    return integrateGaussianLaplace(low, high, center, lsf, fwhm)


@jit
def integrateSwitch(
    low: jnp.ndarray,
    high: jnp.ndarray,
    center: jnp.ndarray,
    lsf: jnp.ndarray,
    fwhm: jnp.ndarray,
    type_idx: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate a single emission line profile selecting the function based on profile_idx.

    Parameters
    ----------
    low : jnp.ndarray
        Low edge of the bin
    high : jnp.ndarray
        High edge of the bin
    center : jnp.ndarray
        Center of the emission line
    lsf : jnp.ndarray
        Line spread function
    fwhm : jnp.ndarray
        Full width at half maximum
    profile_idx : jnp.ndarray
        Index of the profile type

    Returns
    -------
    jnp.ndarray
        Integral across wavelenths
    """
    # map to gaussian, lorentzian, exponential
    branches = (_wrap_gaussian, _wrap_voigt, _wrap_gaussian_laplace)
    return lax.switch(type_idx, branches, low, high, center, lsf, fwhm)


@jit
def integrate(
    low: jnp.ndarray,
    high: jnp.ndarray,
    cent: jnp.ndarray,
    lsf: jnp.ndarray,
    fwhm: jnp.ndarray,
    profile_idx: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate N emission lines over λ bins.
    Returns a matrix of integrals in each bin for each line.
    Uses profile_idx to select the profile type.

    Parameters
    ----------
    low : jnp.ndarray
        Low edge of the bin
    high : jnp.ndarray
        High edge of the bin
    center : jnp.ndarray
        Center of the emission line
    lsf : jnp.ndarray
        Line spread function
    fwfm : jnp.ndarray
        Full width at half maximum
    profile_idx : jnp.ndarray
        Index of the profile type (0=Gaussian, 1=GaussianLaplace, 2=Voigt, etc.)

    Returns
    -------
    jnp.ndarray
        Integral across wavelenths
    """
    # Vectorize the integration across the lines
    vectorized_integrate = vmap(integrateSwitch, in_axes=(None, None, 0, 0, 0, 0))

    # Perform the integration for all lines
    return vectorized_integrate(low, high, cent, lsf, fwhm, profile_idx)


@jit
def linearContinua(
    λ: jnp.ndarray,
    cont_center: jnp.ndarray,
    angles: jnp.ndarray,
    offsets: jnp.ndarray,
    continuum_regions: jnp.ndarray,
    extrapolate: bool = False,
) -> jnp.ndarray:
    """
    Compute the linear model

    Parameters
    ----------
    λ : jnp.ndarray
        Wavelength values
    cont_center : jnp.ndarray
        Centers of the continua
    angles : jnp.ndarray
        Angles of the continua
    offsets : jnp.ndarray
        Offset of the continua
    continuum_regions : jnp.ndarray
        Bounds of the continuum region
    extrapolate : bool
        If True, extrapolate linear segments outside fitted regions.
        If False (default), return 0 outside fitted regions.

    Returns
    -------
    jnp.ndarray
        Flux values
    """

    # Evaluate the linear model
    λ = λ[:, jnp.newaxis]
    continuum = jnp.tan(angles) * (λ - cont_center) + offsets

    # Use lax.cond to handle both extrapolate True/False in a JIT-safe way
    in_region = jnp.logical_and(continuum_regions[:, 0] < λ, λ < continuum_regions[:, 1])
    # If extrapolate=True, return continuum everywhere; if False, mask to region
    return lax.cond(
        extrapolate,
        lambda _: continuum,  # Return continuum everywhere
        lambda _: jnp.where(in_region, continuum, 0.0),  # Mask to regions
        None,  # Operand (unused but required)
    )


@jit
def powerLawContinuum(λ: jnp.ndarray, λ0: float, a: float, β: float) -> jnp.ndarray:
    """
    Compute the power law continuum

    Parameters
    ----------
    λ : jnp.ndarray
        Wavelength values
    λ0 : float
        Reference wavelength
    a : float
        Amplitude of the power law
    β : float
        Power law index

    Returns
    -------
    jnp.ndarray
        Flux values
    """

    return a * ((λ / λ0) ** β)


# Physical constants (SI units)
_H: Final[float] = 6.62607015e-34  # Planck constant (J·s)
_C_SI: Final[float] = 2.99792458e8  # Speed of light (m/s)
_KB: Final[float] = 1.380649e-23  # Boltzmann constant (J/K)


@jit
def _safe_log_expm1(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log(exp(x) - 1) with numerically stable gradients.

    For large x (> ~10): log(exp(x) - 1) ≈ log(exp(x)) = x
    For small x: use log(expm1(x))
    Uses smooth transition to avoid gradient discontinuities.

    Parameters
    ----------
    x : jnp.ndarray
        Input values

    Returns
    -------
    jnp.ndarray
        log(exp(x) - 1) computed with numerical stability
    """
    # For x > 10, log(exp(x) - 1) ≈ x, so use that directly
    # For x <= 10, compute log(expm1(x))
    # Use smooth sigmoid transition around x=10

    # Transition parameter (larger = sharper transition)
    alpha = jax.nn.sigmoid((x - 10.0) / 3.0)

    # For large x: just return x
    large_x_approx = x

    # For small x: compute log(expm1(x)) safely
    # Clip x to avoid overflow in expm1 (expm1 overflows around x=90)
    x_safe = jnp.minimum(x, 50.0)
    small_x_value = jnp.log(jnp.maximum(jnp.expm1(x_safe), 1e-100))

    # Blend between the two
    return jnp.where(x > 50.0, x, alpha * large_x_approx + (1 - alpha) * small_x_value)


@jit
def planck_function(
    wavelength_micron: jnp.ndarray, temperature_k: jnp.ndarray, pivot_micron: float = 0.5
) -> jnp.ndarray:
    """
    Compute normalized Planck function B_λ(T) / B_λ(pivot, T).

    Parameters
    ----------
    wavelength_micron : jnp.ndarray
        REST-FRAME wavelengths in microns
    temperature_k : jnp.ndarray
        Temperature in Kelvin
    pivot_micron : float
        Pivot wavelength for normalization (microns)

    Returns
    -------
    jnp.ndarray
        Normalized Planck function (= 1.0 at pivot wavelength)
    """
    temperature_k = jnp.clip(temperature_k, 20.0, 1e5)
    wavelength_m = wavelength_micron * 1e-6
    pivot_m = pivot_micron * 1e-6

    # Pre-compute constants to avoid gradient overflow
    # Computing (_H * _C_SI) / (wavelength_m * _KB * temperature_k) directly
    # causes -inf gradients in JAX. Instead, pre-compute the constant part.
    const_wave = (_H * _C_SI) / (wavelength_m * _KB)
    const_pivot = (_H * _C_SI) / (pivot_m * _KB)

    x_wave = const_wave / temperature_k
    x_pivot = const_pivot / temperature_k
    x_wave = jnp.clip(x_wave, 0.01, 200.0)
    x_pivot = jnp.clip(x_pivot, 0.01, 200.0)

    log_wavelength_ratio = -5.0 * jnp.log(wavelength_micron / pivot_micron)
    log_expm1_diff = _safe_log_expm1(x_pivot) - _safe_log_expm1(x_wave)

    return jnp.exp(jnp.clip(log_wavelength_ratio + log_expm1_diff, -100.0, 100.0))


@jit
def modified_blackbody(
    wavelength_micron: jnp.ndarray, temperature_k: jnp.ndarray, beta: jnp.ndarray, pivot_micron: float = 1.0
) -> jnp.ndarray:
    """
    Compute normalized modified blackbody: (λ/λ0)^(-β) * B_λ(T) / B_λ(pivot, T).

    Parameters
    ----------
    wavelength_micron : jnp.ndarray
        REST-FRAME wavelengths in microns
    temperature_k : jnp.ndarray
        Temperature in Kelvin
    beta : jnp.ndarray
        Emissivity index (β). β=0 gives pure blackbody
    pivot_micron : float
        Pivot wavelength for normalization (microns)

    Returns
    -------
    jnp.ndarray
        Normalized modified blackbody flux
    """
    beta = jnp.clip(beta, -10.0, 10.0)
    planck = planck_function(wavelength_micron, temperature_k, pivot_micron)
    emissivity = (wavelength_micron / pivot_micron) ** (-beta)
    return emissivity * planck


# V-band reference wavelength for attenuation
_LAMBDA_V: Final[float] = 0.55  # V-band wavelength in microns


@jit
def attenuated_planck(
    wavelength_micron: jnp.ndarray,
    temperature_k: jnp.ndarray,
    tau_v: jnp.ndarray,
    alpha: jnp.ndarray,
    pivot_micron: float = 1.0,
) -> jnp.ndarray:
    """
    Planck blackbody with power-law dust attenuation.

    Parameters
    ----------
    wavelength_micron : array
        REST-FRAME wavelengths in microns.
    temperature_k : float
        Blackbody temperature in Kelvin.
    tau_v : float
        V-band optical depth (at 0.55 micron).
    alpha : float
        Power-law slope for attenuation. Range: -0.4 (Milky Way) to -2.0 (very steep).
        Typical value: -0.7 (LMC-like), -1.6 (SMC-like).
    pivot_micron : float
        Reference wavelength for Planck normalization (default 1.0 micron).

    Returns
    -------
    array
        Attenuated Planck function normalized at pivot wavelength.
    """
    # Clip parameters for numerical stability
    temperature_k = jnp.clip(temperature_k, 20.0, 1e5)
    tau_v = jnp.clip(tau_v, 0.0, 10.0)
    alpha = jnp.clip(alpha, -2.0, 0.0)

    # Planck function (normalized at pivot)
    planck = planck_function(wavelength_micron, temperature_k, pivot_micron)

    # Attenuation: exp(-τ_V × (λ/λ_V)^α), where λ_V = 0.55 micron
    wave_ratio = wavelength_micron / _LAMBDA_V
    attenuation = jnp.exp(-tau_v * jnp.power(wave_ratio, alpha))

    return planck * attenuation
