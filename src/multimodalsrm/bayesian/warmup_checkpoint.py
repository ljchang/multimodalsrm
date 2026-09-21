"""Explicit, runtime-bound warmup checkpoints, separate from fitted model archives."""

import hashlib
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import _archive
from ._backend import runtime
from .posterior_updates import content_id

FORMAT = "personalized-srm-nuts-warmup"


def runtime_identity():
    """Conservative compatibility boundary for short-term exact continuation."""
    jax, _, _, _ = runtime()
    root = Path(__file__).resolve().parents[1]
    return dict(
        runtime=dict(
            python=platform.python_version(),
            machine=platform.machine(),
            packages={n: version(n) for n in ("numpy", "scipy", "jax", "jaxlib", "numpyro")},
            backend=jax.default_backend(),
            devices=[str(d) + ":" + d.device_kind for d in jax.local_devices()],
            x64=bool(jax.config.jax_enable_x64),
            matmul_precision=str(jax.config.jax_default_matmul_precision),
            xla_flags=os.environ.get("XLA_FLAGS", ""),
            prng={
                name: str(jax.config.values[name])
                for name in (
                    "jax_default_prng_impl",
                    "jax_threefry_partitionable",
                    "jax_random_seed_offset",
                    "jax_legacy_prng_key",
                    "jax_enable_custom_prng",
                    "jax_threefry_gpu_kernel_lowering",
                )
            },
        ),
        source={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        },
    )


def _records():
    from numpyro.infer.hmc import HMCState
    from numpyro.infer.hmc_util import HMCAdaptState

    return {c.__name__: c for c in (HMCState, HMCAdaptState)}


def _pack(value):
    if type(value) in _records().values():
        return dict(
            record=type(value).__name__,
            fields={n: _pack(v) for n, v in value._asdict().items()},
        )
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return np.asarray(value)
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_pack(v) for v in value)
    return value


def _unpack(value):
    if isinstance(value, dict):
        if "record" in value:
            cls = _records().get(value["record"])
            if (
                cls is None
                or set(value) != {"record", "fields"}
                or set(value["fields"]) != set(cls._fields)
            ):
                raise ValueError("invalid warmup checkpoint state record")
            return cls(**{k: _unpack(v) for k, v in value["fields"].items()})
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_unpack(v) for v in value)
    return value


class WarmupCheckpoint:
    """Write once, or explicitly restore a checkpoint for an identical training fit."""

    def __init__(self, path, target, *, resume=False):
        self.path = Path(path)
        self.resumed = resume
        self.identity = runtime_identity()
        self.target_id = content_id(target)
        self.saved = None
        if resume:
            self.saved = _archive.read(self.path)
            if (
                not isinstance(self.saved, dict)
                or set(self.saved)
                != {
                    "format",
                    "version",
                    "runtime",
                    "target_id",
                    "space_id",
                    "state",
                    "warmup_seconds",
                }
                or self.saved["format"] != FORMAT
                or self.saved["version"] != 1
            ):
                raise ValueError("invalid warmup checkpoint format")
            if self.saved["target_id"] != self.target_id:
                raise ValueError("warmup checkpoint target identity differs")
            if not _archive.same(self.saved["runtime"], self.identity):
                raise ValueError("warmup checkpoint runtime identity differs")
        elif os.path.lexists(self.path):
            raise FileExistsError(self.path)
        prng = self.identity["runtime"]["prng"]
        if (
            prng["jax_default_prng_impl"] != "threefry2x32"
            or prng["jax_enable_custom_prng"] != "False"
        ):
            raise ValueError("warmup checkpoints require legacy Threefry PRNG keys")

    @staticmethod
    def _space_id(space):
        return content_id(
            dict(
                names=space.problem.names,
                priors=space.problem.parameter_priors,
                active=space.active_indices,
                reference=space.reference,
            )
        )

    def save(self, state, space, seconds):
        self.saved = dict(
            format=FORMAT,
            version=1,
            runtime=self.identity,
            target_id=self.target_id,
            space_id=self._space_id(space),
            state=_pack(state),
            warmup_seconds=float(seconds),
        )
        _archive.write(self.path, self.saved)

    def restore(self, space, config):
        if self.saved["space_id"] != self._space_id(space):
            raise ValueError("warmup checkpoint target parameter identity differs")
        state = _unpack(self.saved["state"])
        if type(state) is not _records()["HMCState"]:
            raise ValueError("invalid warmup checkpoint HMC state")
        prefix = (config.chains,) if config.chains > 1 else ()
        shape = (*prefix, len(space.active_indices))
        if (
            np.shape(state.i) != prefix
            or not np.all(state.i == config.warmup)
            or np.shape(state.z) != shape
            or np.shape(state.z_grad) != shape
            or np.shape(state.rng_key) != (*prefix, 2)
            or np.asarray(state.rng_key).dtype != np.dtype("uint32")
            or np.shape(state.adapt_state.step_size) != prefix
            or not np.all(np.asarray(state.adapt_state.step_size) > 0)
        ):
            raise ValueError("invalid warmup checkpoint state dimensions or iteration")
        jax, jnp, _, _ = runtime()
        state = jax.tree.map(lambda a: jnp.asarray(a), state)
        vg = jax.value_and_grad(space.potential)
        value, grad = jax.jit(jax.vmap(vg))(state.z) if config.chains > 1 else jax.jit(vg)(state.z)
        if not (
            np.allclose(value, state.potential_energy, atol=1e-10, rtol=1e-12)
            and np.allclose(grad, state.z_grad, atol=1e-10, rtol=1e-12)
        ):
            raise ValueError("warmup checkpoint target potential or gradient differs")
        return state

    def metadata(self):
        return dict(
            resumed=self.resumed,
            target_id=self.target_id,
            warmup_seconds=self.saved["warmup_seconds"],
            format=FORMAT,
            version=1,
        )
