import json

import numpy as np
import pandas as pd
import pytest

from midus_stress import pipeline, transforms
from midus_stress.config import CONTINUOUS_OUTCOMES, LPA_INDICATORS
from midus_stress.synthetic import make_synthetic


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(n=800, seed=1)


def test_boxcox_reduces_skew_on_raw_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.lognormal(0, 1.2, 2000)})
    before = df["x"].skew()
    res = transforms.transform_indicators(df, ["x"], "boxcox", skip=())
    assert before > 2
    assert abs(res.data["x"].skew()) < 0.2
    assert res.params["x"]["shift"] == 0.0  # already positive -> no shift


def test_hr_is_left_untransformed_by_default():
    df = pd.DataFrame({"hr": [60.0, 70.0, 80.0], "rmssd": [10.0, 20.0, 40.0]})
    res = transforms.transform_indicators(df, ["hr", "rmssd"], "log")
    assert res.data["hr"].tolist() == [60.0, 70.0, 80.0]
    assert np.allclose(res.data["rmssd"], np.log([10.0, 20.0, 40.0]))


def test_mahalanobis_flags_planted_outlier():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    df.loc[0] = [12, -12, 12]
    flags, cutoff = transforms.mahalanobis_outliers(df, list("abc"))
    assert flags.iloc[0] and cutoff > 0
    assert flags.mean() < 0.06


def test_pipeline_end_to_end(raw):
    result = pipeline.run(raw)
    df = result.data
    assert list(df.columns) == ["patient_id", *LPA_INDICATORS, *CONTINUOUS_OUTCOMES, "cvd_risk"]
    assert not df.isna().any().any()
    assert set(df["cvd_risk"].unique()) <= {0, 1}
    for c in (*LPA_INDICATORS, *CONTINUOUS_OUTCOMES):
        assert abs(df[c].mean()) < 1e-9
        assert df[c].std(ddof=0) == pytest.approx(1.0)
    rows = result.report["rows"]
    assert rows["raw"] >= rows["biomarkers_complete"] >= rows["merged"] > 0
    # Transformed indicators should be close to symmetric.
    for c in LPA_INDICATORS:
        assert abs(result.report["final_distribution"][c]["skewness"]) < 1


def test_no_planted_sentinel_code_survives_cleaning(raw):
    from midus_stress import cleaning
    from midus_stress.config import BIOMARKERS, OUTCOMES
    from midus_stress.synthetic import SENTINEL_CODES
    variables = BIOMARKERS + OUTCOMES
    cleaned = cleaning.apply_missing_rules(cleaning.extract(raw, variables), variables)
    for v in variables:
        code = SENTINEL_CODES[v.raw]
        assert (raw[v.raw] == code).any(), f"fixture should contain {code} in {v.raw}"
        assert not (cleaned[v.name] == code).any(), f"{code} survived in {v.raw}"


def test_pipeline_report_is_json_serialisable(raw):
    json.dumps(pipeline.run(raw).report, default=float)


def test_cli_writes_data_and_report(tmp_path, raw):
    src = tmp_path / "raw.tsv"
    raw.to_csv(src, sep="\t", index=False)
    out = tmp_path / "processed" / "lpa_input.csv"
    pipeline.main(["--input", str(src), "--output", str(out), "--hrv-baseline", "b2"])
    assert out.exists()
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["config"]["hrv_baseline"] == "b2"
    assert report["sentinel_audit"], "audit should list the planted sentinel codes"


def test_winsorize_iqr_clips_only_extremes():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    out, clipped = transforms.winsorize_iqr(df, ["x"], k=3.0)
    assert clipped["x"] == 1
    assert out["x"].iloc[:4].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out["x"].iloc[4] == pytest.approx(4.0 + 3 * 2.0)  # Q3 + 3*IQR (Q1=2, Q3=4)
    assert len(out) == len(df)  # nobody dropped


def test_yeojohnson_only_touches_skewed_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"skewed": rng.lognormal(0, 1, 1000), "normal": rng.normal(0, 1, 1000)})
    res = transforms.transform_indicators(df, ["skewed", "normal"], "yeojohnson", skip=(), skew_threshold=1.0)
    assert "skewed" in res.params and "normal" not in res.params
    assert abs(res.data["skewed"].skew()) < 0.2
    assert res.data["normal"].equals(df["normal"])


@pytest.mark.parametrize("method", ["yeojohnson", "boxcox", "log", "none"])
def test_pipeline_runs_with_each_transform(raw, method):
    result = pipeline.run(raw, pipeline.PipelineConfig(transform=method))
    assert not result.data.isna().any().any()
    assert result.report["config"]["transform"] == method
