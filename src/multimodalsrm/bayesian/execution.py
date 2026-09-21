"""Explicit runtime checks for the requested NUTS chain execution mode."""

from ._backend import runtime


def execution_info(config, *, effective_method=None):
    """Check capacity before sampling and verify the constructed MCMC mode.

    Device enumeration initializes JAX. Configure CPU devices in the process
    environment before calling this function; it never changes global settings.
    """
    jax, _, _, _ = runtime()
    devices = jax.local_devices()
    if config.chain_method == "parallel" and len(devices) < config.chains:
        raise ValueError(
            f"parallel sampling requires {config.chains} local JAX devices; "
            f"found {len(devices)}. Configure devices before starting Python "
            "or explicitly request sequential sampling."
        )
    if effective_method is not None and effective_method != config.chain_method:
        raise RuntimeError(
            f"requested {config.chain_method} chains but NumPyro selected "
            f"{effective_method}; refusing execution fallback"
        )
    return dict(
        requested_chain_method=config.chain_method,
        effective_chain_method=effective_method,
        local_device_count=len(devices),
        backend=jax.default_backend(),
        devices=[str(d) for d in devices],
        chains=config.chains,
        parallel_verified=bool(
            config.chains > 1
            and effective_method == "parallel"
            and config.chain_method == "parallel"
        ),
    )
