"""Generate a fake dataset in the MIDUS 2 column format.

MIDUS data cannot be redistributed, so this lets anyone run the pipeline and
tests end to end. Values are random draws with roughly realistic shapes
(right-skewed HRV power and urinary hormones) and include MIDUS-style sentinel
codes. **They carry no information about real participants.**

    python -m midus_stress.synthetic --n 500 --output data/raw/synthetic_midus.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# One illustrative sentinel code per column, planted in a few rows.
SENTINEL_CODES = {
    "B4VB1LF": 99999, "B4VB2LF": 99998, "B4VB1HF": 99999, "B4VB2HF": 99999,
    "B4VB1RM": 999, "B4VB2RM": 998, "B4VB1HR": 999, "B4VB2HR": 998,
    "B4BCLCRE": 99999, "B4BEPCRE": 99998, "B4BNOCRE": 99999,
    "B4H1A": 8, "B4H1I": 7, "B4H1F": 9, "B4H1B": 8,
    "B4QCESDI": 98, "B4QCESDPA": 98, "B4QCESDDA": 98, "B4QCESDSC": 98,
    "B4QTA_AX": 98, "B4QMA_A": 98, "B4QTA_AG": 99, "B4HOHLTH": 99,
}


def make_synthetic(n: int = 500, seed: int = 0, sentinel_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lognorm = lambda mu, sigma: rng.lognormal(mu, sigma, n).round(2)  # noqa: E731
    df = pd.DataFrame({
        "M2ID": np.arange(10001, 10001 + n),
        "B4VB1LF": lognorm(5.0, 1.0), "B4VB2LF": lognorm(5.0, 1.0),
        "B4VB1HF": lognorm(4.3, 1.2), "B4VB2HF": lognorm(4.3, 1.2),
        "B4VB1RM": lognorm(3.0, 0.6).clip(max=400), "B4VB2RM": lognorm(3.0, 0.6).clip(max=400),
        "B4VB1HR": rng.normal(70, 10, n).round(1).clip(40, 140),
        "B4VB2HR": rng.normal(70, 10, n).round(1).clip(40, 140),
        "B4BCLCRE": lognorm(2.4, 0.6).clip(max=800),
        "B4BEPCRE": lognorm(0.6, 0.6).clip(max=800),
        "B4BNOCRE": lognorm(3.2, 0.4).clip(max=800),
        "B4H1A": rng.choice([1, 2], n, p=[0.1, 0.9]),
        "B4H1I": rng.choice([1, 2], n, p=[0.12, 0.88]),
        "B4H1F": rng.choice([1, 2], n, p=[0.04, 0.96]),
        "B4H1B": rng.choice([1, 2], n, p=[0.35, 0.65]),
        "B4QCESDI": rng.integers(0, 7, n), "B4QCESDPA": rng.integers(0, 13, n),
        "B4QCESDDA": rng.integers(0, 22, n), "B4QCESDSC": rng.integers(0, 22, n),
        "B4QTA_AX": rng.integers(20, 61, n), "B4QMA_A": rng.integers(11, 40, n),
        "B4QTA_AG": rng.integers(10, 35, n), "B4HOHLTH": rng.poisson(1.0, n),
    }).astype({"M2ID": int})

    # Sprinkle in sentinel codes the way the real export uses them.
    for col, code in SENTINEL_CODES.items():
        df[col] = df[col].astype(float)
        df.loc[rng.random(n) < sentinel_rate, col] = code
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="data/raw/synthetic_midus.tsv")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(args.n, args.seed).to_csv(out, sep="\t", index=False)
    print(f"Wrote {args.n} synthetic rows to {out}")


if __name__ == "__main__":
    main()
