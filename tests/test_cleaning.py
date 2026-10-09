import numpy as np
import pandas as pd
import pytest

from midus_stress import cleaning
from midus_stress.config import BIOMARKERS, CVD_ITEMS, OTHER_OUTCOMES, Variable

BY_NAME = {v.name: v for v in BIOMARKERS + CVD_ITEMS + OTHER_OUTCOMES}


def clean(name, values):
    v = BY_NAME[name]
    df = pd.DataFrame({name: values}, dtype=float)
    return cleaning.apply_missing_rules(df, [v])[name].tolist()


def test_real_heart_rates_of_98_and_99_are_kept():
    # Regression: a global missing-code list used to delete these.
    out = clean("baseline_hr_b1", [72.0, 98.0, 99.0, 998.0, 999.0])
    assert out[:3] == [72.0, 98.0, 99.0]
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_biomarker_value_of_7_is_kept():
    # Regression: 7.0 was treated as missing in every column.
    out = clean("urine_cortisol_adj", [7.0, 99999.0])
    assert out[0] == 7.0 and np.isnan(out[1])


def test_wide_codes_removed_from_hrv_power():
    out = clean("low_freq_hrv_b1", [3593.35, 9999.0, 99998.0])
    assert out[0] == 3593.35 and np.isnan(out[1]) and np.isnan(out[2])


def test_categorical_items_keep_only_yes_no():
    out = clean("heart_disease", [1, 2, 3, 7, 8, 9])
    assert out[:2] == [1.0, 2.0]
    assert all(np.isnan(x) for x in out[2:])


def test_scale_scores_outside_range_are_missing():
    out = clean("anger", [10.0, 27.0, 40.0, 98.0, 99.0])
    assert out[:3] == [10.0, 27.0, 40.0]
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_depression_total_is_missing_if_any_subscale_missing():
    # Regression: pandas sum() treats NaN as 0, giving a score of 0.
    df = pd.DataFrame({
        "depression_i": [1, np.nan, np.nan],
        "depression_pa": [2, 3, np.nan],
        "depression_da": [3, 4, np.nan],
        "depression_sc": [4, 5, np.nan],
    }, dtype=float)
    total = cleaning.depression_total(df)
    assert total.iloc[0] == 10
    assert total.iloc[1:].isna().all()


def test_cvd_risk_unknown_when_no_yes_and_some_missing():
    # Regression: missing answers were counted as 'no risk'.
    df = pd.DataFrame({
        "heart_disease": [1, 2, 2, np.nan],
        "diabetes":      [2, 2, np.nan, np.nan],
        "stroke":        [2, 2, 2, 1],
        "high_bp":       [2, 2, 2, np.nan],
    })
    risk = cleaning.cvd_risk(df)
    assert risk.iloc[0] == 1      # one 'yes'
    assert risk.iloc[1] == 0      # all 'no'
    assert np.isnan(risk.iloc[2])  # 'no' x3 + missing -> unknown
    assert risk.iloc[3] == 1      # a 'yes' decides it even with gaps


@pytest.mark.parametrize("strategy, expected", [("mean", 15.0), ("b1", 10.0), ("b2", 20.0)])
def test_combine_hrv_baselines(strategy, expected):
    df = pd.DataFrame({
        "low_freq_hrv_b1": [10.0], "low_freq_hrv_b2": [20.0],
        "high_freq_hrv_b1": [1.0], "high_freq_hrv_b2": [1.0],
        "rmssd_hrv_b1": [1.0], "rmssd_hrv_b2": [1.0],
        "baseline_hr_b1": [60.0], "baseline_hr_b2": [60.0],
    })
    out = cleaning.combine_hrv_baselines(df, strategy)
    assert out["lf_hrv"].iloc[0] == expected
    assert "low_freq_hrv_b1" not in out.columns


def test_combine_mean_falls_back_to_available_baseline():
    df = pd.DataFrame({
        "low_freq_hrv_b1": [np.nan], "low_freq_hrv_b2": [20.0],
        "high_freq_hrv_b1": [1.0], "high_freq_hrv_b2": [1.0],
        "rmssd_hrv_b1": [1.0], "rmssd_hrv_b2": [1.0],
        "baseline_hr_b1": [60.0], "baseline_hr_b2": [60.0],
    })
    assert cleaning.combine_hrv_baselines(df)["lf_hrv"].iloc[0] == 20.0


def test_audit_reports_codes_and_how_they_are_treated():
    raw = pd.DataFrame({"M2ID": [1, 2, 3], "B4VB1HR": [98.0, 999.0, 70.0]})
    v = BY_NAME["baseline_hr_b1"]
    audit = cleaning.audit_sentinels(raw, [v]).set_index("code")
    assert bool(audit.loc[98.0, "treated_as_missing"]) is False
    assert bool(audit.loc[999.0, "treated_as_missing"]) is True


def test_extract_raises_on_missing_columns():
    with pytest.raises(KeyError):
        cleaning.extract(pd.DataFrame({"M2ID": [1]}), [Variable("NOPE", "nope", "")])
