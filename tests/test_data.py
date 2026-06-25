"""ML-specific CI: PSI drift check between reference and current data."""

import os
import numpy as np
import pandas as pd
import pytest
from src.data import load_data


PSI_THRESHOLD = 0.2
N_BINS = 10
# Check drift on the first 5 features (most informative for breast cancer)
FEATURES_TO_CHECK = load_data(inject_drift=False).drop(columns=["target"]).columns[:5].tolist()


def compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = N_BINS) -> float:
    """Compute Population Stability Index between two distributions."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Avoid division by zero
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def test_no_data_drift():
    """Fail if PSI exceeds threshold on any monitored feature."""
    reference_path = "data/reference_data.csv"
    inject_drift = os.environ.get("INJECT_DRIFT", "false").lower() == "true"

    reference_df = pd.read_csv(reference_path)
    current_df = load_data(inject_drift=inject_drift)

    drifted_features = []
    for feature in FEATURES_TO_CHECK:
        psi = compute_psi(reference_df[feature], current_df[feature])
        print(f"PSI [{feature}]: {psi:.4f}")
        if psi > PSI_THRESHOLD:
            drifted_features.append((feature, psi))

    assert not drifted_features, (
        f"Data drift detected (PSI > {PSI_THRESHOLD}) in features: "
        + ", ".join(f"{f}={v:.4f}" for f, v in drifted_features)
    )
