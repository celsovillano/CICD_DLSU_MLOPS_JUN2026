"""Load breast cancer data, optionally inject drift, and save as CSV."""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data(inject_drift: bool = False) -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    if inject_drift:
        # Shift the top 5 features to simulate distribution drift
        drift_cols = df.columns[:5].tolist()
        df[drift_cols] = df[drift_cols] * 1.8 + df[drift_cols].std() * 2.5

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    inject = os.environ.get("INJECT_DRIFT", "false").lower() == "true"
    df = load_data(inject_drift=inject)

    output_path = "data/current_data.csv"
    save_data(df, output_path)
    print(f"Saved {'drifted' if inject else 'clean'} data to {output_path}")
