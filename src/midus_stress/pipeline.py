"""End-to-end preprocessing: raw MIDUS 2 table -> analysis-ready LPA dataset.

Usage::

    python -m midus_stress.pipeline --input data/raw/midus2_agg_data.tsv \
        --output data/processed/lpa_input.csv

Writes the processed dataset plus a JSON report next to it. The report holds
only aggregate information (row counts, transform parameters, distribution
summaries, sentinel-code audit), so it is safe to commit or share.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from . import cleaning, transforms
from .config import (
    BIOMARKERS,
    CONTINUOUS_OUTCOMES,
    CVD_ITEMS,
    CESD_SUBSCALES,
    ID_NAME,
    LPA_INDICATORS,
    OUTCOMES,
)


@dataclass
class PipelineConfig:
    """Defaults reproduce the preprocessing behind the SPARC 2026 results:
    average both baselines, clip at 3 x IQR, Yeo-Johnson any indicator with
    |skew| > 1, then z-score. ``transform="boxcox"`` gives the alternative
    explored in notebook 03.
    """

    hrv_baseline: cleaning.HrvBaseline = "mean"
    winsorize_iqr: Optional[float] = 3.0
    transform: transforms.Transform = "yeojohnson"
    skew_threshold: float = 1.0
    standardize_outcomes: bool = True


@dataclass
class PipelineResult:
    data: pd.DataFrame
    report: Dict[str, Any] = field(default_factory=dict)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the aggregated MIDUS 2 export (TSV or CSV)."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def run(raw: pd.DataFrame, config: Optional[PipelineConfig] = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    report: Dict[str, Any] = {"config": cfg.__dict__.copy(), "rows": {"raw": int(len(raw))}}

    # 1. Biomarkers -----------------------------------------------------------
    bio = cleaning.extract(raw, BIOMARKERS)
    bio = cleaning.apply_missing_rules(bio, BIOMARKERS)
    report["missing_after_rules"] = {"biomarkers": bio.drop(columns=ID_NAME).isna().sum().astype(int).to_dict()}
    bio = cleaning.combine_hrv_baselines(bio, cfg.hrv_baseline)
    bio = cleaning.drop_incomplete(bio)
    report["rows"]["biomarkers_complete"] = int(len(bio))

    # 2. Outcomes -------------------------------------------------------------
    out = cleaning.extract(raw, OUTCOMES)
    out = cleaning.apply_missing_rules(out, OUTCOMES)
    report["missing_after_rules"]["outcomes"] = out.drop(columns=ID_NAME).isna().sum().astype(int).to_dict()
    out["depression"] = cleaning.depression_total(out)
    out["cvd_risk"] = cleaning.cvd_risk(out)
    out = out.drop(columns=[v.name for v in CESD_SUBSCALES + CVD_ITEMS])
    out = cleaning.drop_incomplete(out)
    out["cvd_risk"] = out["cvd_risk"].astype(int)
    report["rows"]["outcomes_complete"] = int(len(out))

    # 3. Merge ----------------------------------------------------------------
    merged = bio.merge(out, on=ID_NAME, how="inner")
    report["rows"]["merged"] = int(len(merged))
    report["cvd_risk_counts"] = merged["cvd_risk"].value_counts().sort_index().to_dict()
    report["raw_distribution"] = transforms.distribution_summary(merged, LPA_INDICATORS).round(3).to_dict(orient="index")

    # 4. Clip extremes, transform raw indicators, then standardise -----------
    if cfg.winsorize_iqr is not None:
        merged, clipped = transforms.winsorize_iqr(merged, LPA_INDICATORS, cfg.winsorize_iqr)
        report["winsorized_values"] = clipped
    skip = ("hr",) if cfg.transform == "boxcox" else ()
    tr = transforms.transform_indicators(merged, LPA_INDICATORS, cfg.transform,
                                         skip=skip, skew_threshold=cfg.skew_threshold)
    report["transform_params"] = tr.params
    final = transforms.zscore(tr.data, LPA_INDICATORS)
    if cfg.standardize_outcomes:
        final = transforms.zscore(final, CONTINUOUS_OUTCOMES)
    report["final_distribution"] = transforms.distribution_summary(final, LPA_INDICATORS).round(3).to_dict(orient="index")

    columns = [ID_NAME, *LPA_INDICATORS, *CONTINUOUS_OUTCOMES, "cvd_risk"]
    return PipelineResult(final[columns].reset_index(drop=True), report)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Raw MIDUS 2 aggregated export (.tsv or .csv)")
    p.add_argument("--output", required=True, help="Where to write the processed CSV")
    p.add_argument("--hrv-baseline", choices=["mean", "b1", "b2"], default="mean")
    p.add_argument("--transform", choices=["none", "log", "boxcox", "yeojohnson"], default="yeojohnson")
    p.add_argument("--skew-threshold", type=float, default=1.0,
                   help="Only transform indicators with |skew| above this (default 1.0)")
    p.add_argument("--winsorize-iqr", type=float, default=3.0,
                   help="Clip at Q1/Q3 -/+ k*IQR; pass 0 to disable (default 3)")
    p.add_argument("--no-standardize-outcomes", action="store_true")
    args = p.parse_args(argv)

    raw = load_raw(args.input)
    cfg = PipelineConfig(
        hrv_baseline=args.hrv_baseline,
        winsorize_iqr=args.winsorize_iqr or None,
        transform=args.transform,
        skew_threshold=args.skew_threshold,
        standardize_outcomes=not args.no_standardize_outcomes,
    )
    result = run(raw, cfg)
    result.report["sentinel_audit"] = cleaning.audit_sentinels(raw, BIOMARKERS + OUTCOMES).to_dict(orient="records")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(out_path, index=False)
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(result.report, indent=2, default=float))

    rows = result.report["rows"]
    print(f"Raw rows: {rows['raw']} | biomarkers complete: {rows['biomarkers_complete']} | "
          f"outcomes complete: {rows['outcomes_complete']} | final: {rows['merged']}")
    print(f"Wrote {out_path} and {report_path}")


if __name__ == "__main__":
    main()
