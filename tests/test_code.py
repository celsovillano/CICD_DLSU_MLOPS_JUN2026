"""Classic CI: basic unit tests for src modules."""

import pandas as pd
from src.data import load_data
from src.train import train


def test_load_data_returns_dataframe():
    df = load_data(inject_drift=False)
    assert isinstance(df, pd.DataFrame)


def test_load_data_has_target_column():
    df = load_data(inject_drift=False)
    assert "target" in df.columns


def test_load_data_no_nulls():
    df = load_data(inject_drift=False)
    assert df.isnull().sum().sum() == 0


def test_train_returns_pipeline():
    from sklearn.pipeline import Pipeline
    df = load_data(inject_drift=False)
    model = train(df)
    assert isinstance(model, Pipeline)


def test_drift_flag_changes_data():
    clean = load_data(inject_drift=False)
    drifted = load_data(inject_drift=True)
    assert not clean.drop(columns=["target"]).equals(drifted.drop(columns=["target"]))
