from typing import Optional

import polars as pl

from ._rustystats import (
    _norm_cdf,
    _norm_ppf,
    _poisson,
    _poisson_repeat_indices,
)


class norm:
    """Functions for working with a normal continuous random variable."""

    @staticmethod
    def ppf(q: float) -> float:
        r"""The percent point function. Also called the quantile, percentile, inverse
        CDF, or inverse distribution function. Computes the value of a random variable
        such that its probability is \( \leq q \). If `q` is 0, it returns negative
        infinity, if `q` is 1, it returns infinity. Any number outside of [0, 1] will
        result in NaN.

        Parameters
        ----------
        q : float
            Probability value

        Returns
        -------
        float
            Likelihood a random variable is realized in the range at or below `q` for
            the normal distribution.

        Added in version 0.0.24
        -----------------------
        """
        return _norm_ppf(q)

    @staticmethod
    def cdf(x: float) -> float:
        r"""The cumulative distribution function.

        Parameters
        ----------
        x : float

        Returns
        -------
        float
            The probability a random variable will take a value \( \leq x \)

        Added in version 0.0.24
        -----------------------
        """
        return _norm_cdf(x)


class Random:
    def __init__(self, seed: Optional[int]):
        self.seed = seed

    def _increment_seed(self):
        if self.seed is not None:
            self.seed += 1

    def poisson(self, lam: float, size: int) -> list[int]:
        res = _poisson(lam=lam, size=size, seed=self.seed)

        self._increment_seed()

        return res

    def poisson_repeat_indices(self, lam: float, size: int) -> list[int]:
        """Poisson(`lam`) draw counts expanded into gather indices.

        Row `i` appears `count[i]` times, in order -- the resample that
        `Random.poisson` counts describe, without building the counts in Python.
        """
        res = _poisson_repeat_indices(lam=lam, size=size, seed=self.seed)

        self._increment_seed()

        return res


def _pl_norm_ppf(expr: pl.Expr) -> pl.Expr:
    """Vectorised `norm.ppf` over a column.

    A plugin expression rather than `map_elements(norm.ppf)`: a Python UDF runs one call
    per element while holding the GIL, which is exactly the serialisation the extension
    was changed to avoid.
    """
    from ._polars._utils import _PLUGIN_PATH

    return pl.plugins.register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="pl_norm_ppf",
        args=[expr.cast(pl.Float64)],
        is_elementwise=True,
    )


def _pl_norm_cdf(expr: pl.Expr) -> pl.Expr:
    """Vectorised `norm.cdf` over a column. See `_pl_norm_ppf`."""
    from ._polars._utils import _PLUGIN_PATH

    return pl.plugins.register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="pl_norm_cdf",
        args=[expr.cast(pl.Float64)],
        is_elementwise=True,
    )
