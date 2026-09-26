#!/usr/bin/env python3
"""Bounded native thread-pool caps for the parallel pilot.

This module must stay dependency-free and must be imported *before* NumPy,
because the OpenMP/BLAS runtimes read their pool size once at first use.  Once
NumPy is loaded, setting these variables has no effect.

The pilot fans work out across worker processes and every per-worker numeric
kernel is a 30-wide dot product.  Letting each worker also open a native thread
pool oversubscribes the CPU (10 workers x N BLAS threads) and measurably slows
the run, so each worker is pinned to a single native thread by default.

Values already present in the environment win, so an operator can still override
a single knob from the shell.
"""

from __future__ import annotations

import os

# Every runtime that might otherwise spawn its own pool inside each worker.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",        # older OpenBLAS builds
)


def cap_native_threads(threads: int = 1) -> dict[str, str]:
    """Pin native math runtimes to ``threads`` per process.

    Returns the resulting mapping so the caller can log exactly what the workers
    will see.  Existing environment values are preserved, not overwritten.
    """
    if threads < 1:
        raise ValueError("threads must be >= 1")
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, str(threads))
    return {name: os.environ[name] for name in THREAD_ENV_VARS}
