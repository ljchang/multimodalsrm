"""Ordered process tasks with one active level of parallelism."""

import sys
import warnings
from contextvars import ContextVar

import numpy as np
from joblib import Parallel, cpu_count, delayed, parallel_config

_worker = ContextVar("multimodal_parallel_worker", default=False)


def worker_count(n_jobs, tasks):
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or (n_jobs != -1 and n_jobs < 1)
    ):
        raise ValueError("n_jobs must be a positive integer or -1")
    if _worker.get():
        return 1
    return max(1, min(cpu_count() if n_jobs == -1 else int(n_jobs), tasks))


def _run_worker(function, arguments, warning_filters):
    token = _worker.set(True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            # catch_warnings invalidates warning registries on entry. Install
            # the caller's filters before any task code can issue a warning.
            warnings.filters[:] = warning_filters
            result = function(*arguments)
        modules = {
            getattr(module, "__file__", None): name for name, module in list(sys.modules.items())
        }
        return result, [
            (w.message, w.category, w.filename, w.lineno, modules.get(w.filename)) for w in caught
        ]
    finally:
        _worker.reset(token)


def ordered_tasks(function, arguments, n_jobs):
    """Run tasks in isolated processes; preserve input order and failures.

    Inside our workers, downstream fold/restart requests run serially. Loky's
    inner thread limit covers OMP, OpenBLAS, MKL and Accelerate environment
    settings, without changing the parent process environment.
    """
    arguments = list(arguments)
    workers = worker_count(n_jobs, len(arguments))
    if workers == 1:
        return [function(*args) for args in arguments]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        completed = Parallel(
            n_jobs=workers,
            pre_dispatch=workers,
            batch_size=1,
            max_nbytes="1M",
            mmap_mode="r",
        )(delayed(_run_worker)(function, args, list(warnings.filters)) for args in arguments)
    results = []
    for result, caught in completed:
        for message, category, filename, lineno, module in caught:
            kwargs = {}
            if module is not None:
                kwargs["module"] = module
                if module in sys.modules:
                    kwargs["registry"] = vars(sys.modules[module]).setdefault(
                        "__warningregistry__", {}
                    )
            warnings.warn_explicit(message, category, filename, lineno, **kwargs)
        results.append(result)
    return results
