"""Distribution transforms and outlier diagnostics for the LPA indicators.

Latent profile analysis with Gaussian mixtures assumes roughly normal
indicators within each profile. Raw HRV power and urinary hormone values are
heavily right-skewed (skew of 2-8), so they are transformed before modelling.
See notebooks/03 for the comparison that led to Box-Cox as the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

Transform = Literal["none", "log", "boxcox"]


@dataclass
class TransformResult:
    data: pd.DataFrame
    method: Transform
    # Box-Cox lambda and shift per variable, kept so the transform is reproducible.
    params: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _positive_shift(s: pd.Series) -> float:
    """Shift needed so the minimum value is at least 1 (0 if already positive)."""
    m = s.min()
    return 0.0 if m > 0 else float(1.0 - m)


def transform_indicators(df: pd.DataFrame, columns: Sequence[str], method: Transform = "boxcox",
                         skip: Sequence[str] = ("hr",)) -> TransformResult:
    """Apply ``method`` to raw (unstandardised) ``columns``.

    Transforms must run on raw values, *before* z-scoring. Shifting z-scores to
    be positive and then logging them (as an earlier exploratory notebook did)
    is not equivalent to log-transforming the measurement.

    Heart rate is roughly normal already and is skipped by default.
    """
    out = df.copy()
    params: Dict[str, Dict[str, float]] = {}
    if method == "none":
        return TransformResult(out, method, params)
    for col in columns:
        if col in skip:
            continue
        s = out[col].dropna()
        shift = _positive_shift(s)
        if method == "log":
            out.loc[s.index, col] = np.log(s + shift)
            params[col] = {"shift": shift}
        elif method == "boxcox":
            values, lam = stats.boxcox(s + shift)
            out.loc[s.index, col] = values
            params[col] = {"lambda": float(lam), "shift": shift}
        else:
            raise ValueError(f"Unknown transform: {method!r}")
    return TransformResult(out, method, params)


def zscore(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Standardise ``columns`` to mean 0, SD 1 (population SD, like sklearn)."""
    out = df.copy()
    sub = out[list(columns)].astype(float)
    out[list(columns)] = (sub - sub.mean()) / sub.std(ddof=0)
    return out


def mahalanobis_outliers(df: pd.DataFrame, columns: Sequence[str],
                         quantile: float = 0.975) -> Tuple[pd.Series, float]:
    """Flag multivariate outliers whose squared Mahalanobis distance exceeds
    the chi-squared ``quantile`` with ``len(columns)`` degrees of freedom.

    Returns (boolean Series aligned to ``df``, chi-squared cutoff).
    Rows with any missing indicator are never flagged.
    """
    x = df[list(columns)].dropna()
    cov = np.cov(x.values, rowvar=False)
    inv = np.linalg.pinv(cov)
    centered = x.values - x.values.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", centered, inv, centered)
    cutoff = float(stats.chi2.ppf(quantile, df=len(columns)))
    flags = pd.Series(False, index=df.index)
    flags.loc[x.index] = d2 > cutoff
    return flags, cutoff


def distribution_summary(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Skewness, kurtosis and IQR-rule outlier counts per column."""
    rows = []
    for c in columns:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append({
            "variable": c,
            "n": int(s.size),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
        })
    return pd.DataFrame(rows).set_index("variable")
