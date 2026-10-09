"""Cleaning steps: extract variables, apply missing-data rules, derive scores."""

from __future__ import annotations

from typing import Iterable, List, Literal, Sequence

import numpy as np
import pandas as pd

from .config import (
    CANDIDATE_SENTINELS,
    CESD_SUBSCALES,
    CVD_ITEMS,
    HRV_PAIRS,
    ID_COLUMN,
    ID_NAME,
    Variable,
)

HrvBaseline = Literal["mean", "b1", "b2"]


def extract(raw: pd.DataFrame, variables: Sequence[Variable]) -> pd.DataFrame:
    """Select the ID plus ``variables`` from the raw MIDUS table and rename them."""
    wanted = [ID_COLUMN] + [v.raw for v in variables]
    missing = [c for c in wanted if c not in raw.columns]
    if missing:
        raise KeyError(f"Columns not found in input data: {missing}")
    out = raw[wanted].rename(columns={ID_COLUMN: ID_NAME, **{v.raw: v.name for v in variables}})
    return out.apply(lambda s: s if s.name == ID_NAME else pd.to_numeric(s, errors="coerce"))


def apply_missing_rules(df: pd.DataFrame, variables: Sequence[Variable]) -> pd.DataFrame:
    """Set values to NaN according to each variable's own validity rule."""
    out = df.copy()
    for v in variables:
        col = out[v.name]
        invalid = col.isin(v.missing_codes)
        low, high = v.valid_range
        if low is not None:
            invalid |= col < low
        if high is not None:
            invalid |= col > high
        if v.allowed_values is not None:
            invalid |= col.notna() & ~col.isin(v.allowed_values)
        out[v.name] = col.mask(invalid)
    return out


def audit_sentinels(raw: pd.DataFrame, variables: Iterable[Variable]) -> pd.DataFrame:
    """Count how often each candidate sentinel code appears in each raw column.

    Use this against the real data file to confirm the rules in ``config.py``:
    any code listed here that is *not* removed by a variable's rule will be
    kept as a real value.
    """
    rows = []
    for v in variables:
        if v.raw not in raw.columns:
            continue
        col = pd.to_numeric(raw[v.raw], errors="coerce")
        for code in CANDIDATE_SENTINELS:
            n = int((col == code).sum())
            if n:
                removed = bool(apply_missing_rules(pd.DataFrame({v.name: [code]}), [v])[v.name].isna().iloc[0])
                rows.append({"variable": v.raw, "name": v.name, "code": code, "count": n, "treated_as_missing": removed})
    return pd.DataFrame(rows, columns=["variable", "name", "code", "count", "treated_as_missing"])


def combine_hrv_baselines(df: pd.DataFrame, strategy: HrvBaseline = "mean") -> pd.DataFrame:
    """Collapse the two resting baselines (B1, B2) into one column per HRV measure.

    ``"mean"`` averages B1 and B2, falling back to whichever one is present.
    ``"b1"`` / ``"b2"`` keep a single baseline period.
    """
    if strategy not in ("mean", "b1", "b2"):
        raise ValueError(f"Unknown HRV baseline strategy: {strategy!r}")
    out = df.copy()
    drop: List[str] = []
    for new, (b1, b2) in HRV_PAIRS.items():
        if strategy == "mean":
            out[new] = out[[b1, b2]].mean(axis=1, skipna=True)
        else:
            out[new] = out[b1 if strategy == "b1" else b2]
        drop += [b1, b2]
    return out.drop(columns=drop)


def depression_total(df: pd.DataFrame) -> pd.Series:
    """Sum the four CES-D subscales. Missing if *any* subscale is missing.

    ``DataFrame.sum`` treats NaN as 0 by default, which would give someone
    with no CES-D data a depression score of 0. ``min_count`` prevents that.
    """
    cols = [v.name for v in CESD_SUBSCALES]
    return df[cols].sum(axis=1, min_count=len(cols))


def cvd_risk(df: pd.DataFrame) -> pd.Series:
    """1 if any CVD condition is reported, 0 if all four are 'no', else missing.

    Someone who answered 'no' to two items and left two blank has unknown risk,
    not zero risk, so they are kept as NaN rather than counted as 0.
    """
    items = df[[v.name for v in CVD_ITEMS]]
    any_yes = (items == 1).any(axis=1)
    all_no = (items == 2).all(axis=1)
    return pd.Series(np.select([any_yes, all_no], [1.0, 0.0], default=np.nan), index=df.index, name="cvd_risk")


def drop_incomplete(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Listwise deletion on ``columns`` (default: all columns)."""
    return df.dropna(subset=list(columns) if columns is not None else None).copy()
