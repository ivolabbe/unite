import os, sys

if "jax" in sys.modules:
    raise RuntimeError(
        "JAX_ENABLE_X64 must be set before importing jax. "
        "Set the environment variable or import this package first."
        'os.environ.setdefault("JAX_ENABLE_X64", "1")'
    )

os.environ.setdefault("JAX_ENABLE_X64", "1")
