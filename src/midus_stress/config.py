"""Variable mappings and data-quality rules for the MIDUS 2 Biomarker Project.

Variable names follow the ICPSR 29282 release (MIDUS 2 Biomarker Project,
2004-2009). Each analysis variable gets a readable name and an explicit
validity rule, so missing-data handling is decided per column rather than by
one global list of codes.

Why per-column rules?
    An earlier version of the pipeline replaced the values 7, 98, 99, ... with
    NaN in *every* column. That silently deleted real observations such as a
    resting heart rate of 98 bpm or a cortisol value of 7.0 ug/g. MIDUS uses
    sentinel codes whose width matches the field (e.g. 8/9 for 1-digit
    categorical items, 99998/99999 for wide numeric fields), so a value is only
    a sentinel in the context of its own column.

NOTE: the ranges below are deliberately wide plausibility bounds chosen so that
real values survive and the sentinel codes fall outside them. They should be
confirmed against the ICPSR 29282 codebook (login required). Run
``midus_stress.cleaning.audit_sentinels`` on the raw file to see exactly which
candidate codes occur in each column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

ID_COLUMN = "M2ID"
ID_NAME = "patient_id"


@dataclass(frozen=True)
class Variable:
    """A raw MIDUS variable, its analysis name, and how to recognise missing data."""

    raw: str
    name: str
    description: str
    # Values outside [low, high] are treated as missing. None = unbounded.
    valid_range: Tuple[Optional[float], Optional[float]] = (None, None)
    # Explicit sentinel codes, removed even if they fall inside valid_range.
    missing_codes: FrozenSet[float] = field(default_factory=frozenset)
    # For categorical items: the only values that count as real answers.
    allowed_values: Optional[FrozenSet[float]] = None


WIDE_NUMERIC_CODES = frozenset({9998.0, 9999.0, 99998.0, 99999.0})

# --------------------------------------------------------------------------- #
# LPA indicators: baseline (resting) physiology
# B1 / B2 are the two resting baseline periods of the psychophysiology protocol.
# --------------------------------------------------------------------------- #
HRV_POWER = dict(valid_range=(0.0, None), missing_codes=WIDE_NUMERIC_CODES)

BIOMARKERS: Tuple[Variable, ...] = (
    Variable("B4VB1LF", "low_freq_hrv_b1", "LF-HRV power, baseline 1 (ms^2)", **HRV_POWER),
    Variable("B4VB2LF", "low_freq_hrv_b2", "LF-HRV power, baseline 2 (ms^2)", **HRV_POWER),
    Variable("B4VB1HF", "high_freq_hrv_b1", "HF-HRV power, baseline 1 (ms^2)", **HRV_POWER),
    Variable("B4VB2HF", "high_freq_hrv_b2", "HF-HRV power, baseline 2 (ms^2)", **HRV_POWER),
    Variable("B4VB1RM", "rmssd_hrv_b1", "RMSSD, baseline 1 (ms)", valid_range=(0.0, 500.0)),
    Variable("B4VB2RM", "rmssd_hrv_b2", "RMSSD, baseline 2 (ms)", valid_range=(0.0, 500.0)),
    Variable("B4VB1HR", "baseline_hr_b1", "Heart rate, baseline 1 (bpm)", valid_range=(25.0, 220.0)),
    Variable("B4VB2HR", "baseline_hr_b2", "Heart rate, baseline 2 (bpm)", valid_range=(25.0, 220.0)),
    Variable("B4BCLCRE", "urine_cortisol_adj", "12-h urine cortisol / creatinine (ug/g)", valid_range=(0.0, 900.0)),
    Variable("B4BEPCRE", "urine_epinephrine_adj", "12-h urine epinephrine / creatinine (ug/g)", valid_range=(0.0, 900.0)),
    Variable("B4BNOCRE", "urine_norepinephrine_adj", "12-h urine norepinephrine / creatinine (ug/g)", valid_range=(0.0, 900.0)),
)

# The two baseline periods are combined into one value per HRV measure.
HRV_PAIRS: Dict[str, Tuple[str, str]] = {
    "lf_hrv": ("low_freq_hrv_b1", "low_freq_hrv_b2"),
    "hf_hrv": ("high_freq_hrv_b1", "high_freq_hrv_b2"),
    "rmssd": ("rmssd_hrv_b1", "rmssd_hrv_b2"),
    "hr": ("baseline_hr_b1", "baseline_hr_b2"),
}

# Final LPA indicator set (7 variables across 3 stress systems).
LPA_INDICATORS: Tuple[str, ...] = (
    "lf_hrv",
    "hf_hrv",
    "rmssd",
    "hr",
    "urine_cortisol_adj",
    "urine_epinephrine_adj",
    "urine_norepinephrine_adj",
)

# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #
YES_NO = dict(allowed_values=frozenset({1.0, 2.0}))  # 1 = yes, 2 = no

CVD_ITEMS: Tuple[Variable, ...] = (
    Variable("B4H1A", "heart_disease", "Ever had heart disease (1=yes, 2=no)", **YES_NO),
    Variable("B4H1I", "diabetes", "Ever had diabetes (1=yes, 2=no)", **YES_NO),
    Variable("B4H1F", "stroke", "Ever had a stroke (1=yes, 2=no)", **YES_NO),
    Variable("B4H1B", "high_bp", "Ever had high blood pressure (1=yes, 2=no)", **YES_NO),
)

CESD_SUBSCALES: Tuple[Variable, ...] = (
    Variable("B4QCESDI", "depression_i", "CES-D interpersonal subscale", valid_range=(0.0, 60.0)),
    Variable("B4QCESDPA", "depression_pa", "CES-D positive affect subscale", valid_range=(0.0, 60.0)),
    Variable("B4QCESDDA", "depression_da", "CES-D depressed affect subscale", valid_range=(0.0, 60.0)),
    Variable("B4QCESDSC", "depression_sc", "CES-D somatic complaints subscale", valid_range=(0.0, 60.0)),
)

OTHER_OUTCOMES: Tuple[Variable, ...] = (
    Variable("B4QTA_AX", "anxiety", "Spielberger trait anxiety", valid_range=(10.0, 80.0)),
    Variable("B4QMA_A", "distress_anxious", "MASQ general distress: anxious symptoms", valid_range=(11.0, 55.0)),
    Variable("B4QTA_AG", "anger", "Spielberger trait anger", valid_range=(10.0, 40.0)),
    Variable("B4HOHLTH", "phys_hlth_evnts_tot", "Count of other major health events", valid_range=(0.0, 50.0)),
)

OUTCOMES: Tuple[Variable, ...] = CVD_ITEMS + CESD_SUBSCALES + OTHER_OUTCOMES

# Continuous outcome columns produced by the pipeline (cvd_risk is binary).
CONTINUOUS_OUTCOMES: Tuple[str, ...] = (
    "anxiety",
    "distress_anxious",
    "anger",
    "phys_hlth_evnts_tot",
    "depression",
)

# --------------------------------------------------------------------------- #
# Covariates (mapped, not yet cleaned by this package - see README roadmap)
# --------------------------------------------------------------------------- #
COVARIATE_COLUMN_MAP: Dict[str, str] = {
    "B1PRAGE_2019": "age",
    "B1SG8A": "income",
    "B1PB1": "education",
    "B1SA37B": "height_in",
    "B1SA39": "weight_lb",
    "B1PA38A": "ever_smoked",
    "B1PA39": "now_smokes",
    "B4H25": "phys_active",
    "B1PF7A": "race",
}

# Candidate sentinel codes reported by the audit helper.
CANDIDATE_SENTINELS: Tuple[float, ...] = (
    7.0, 8.0, 9.0, 97.0, 98.0, 99.0, 997.0, 998.0, 999.0,
    9997.0, 9998.0, 9999.0, 99997.0, 99998.0, 99999.0,
)
