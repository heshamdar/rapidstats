"""Reference helpers shared by tests.

These are deliberately slow and obvious. They exist to be believed, not to be fast --
the library's fast paths are asserted against them.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def materialised_resample(df: pl.DataFrame, counts) -> pl.DataFrame:
    """Expand `df` by repeating row `i` exactly `counts[i]` times.

    This is the explicit, unambiguous meaning of a bootstrap resample. The library's
    weight-based fast path multiplies `sample_weight` by these same counts instead of
    materialising the rows; tests assert the two agree.
    """
    counts = np.asarray(counts)

    if counts.shape[0] != df.height:
        raise ValueError(
            f"`counts` has length {counts.shape[0]} but frame has height {df.height}"
        )

    if (counts < 0).any():
        raise ValueError("`counts` must be non-negative")

    return df[np.repeat(np.arange(df.height), counts)]


def poisson_counts(n: int, seed: int, lam: float = 1.0) -> np.ndarray:
    """Poisson(lam) resample multiplicities -- the 'poisson' sampling method."""
    return np.random.default_rng(seed).poisson(lam, n)


def multinomial_counts(n: int, seed: int) -> np.ndarray:
    """Multinomial resample multiplicities -- sampling `n` rows with replacement."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n))
