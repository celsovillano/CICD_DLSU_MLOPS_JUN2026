"""Train a logistic regression model on breast cancer data and save it."""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


MODEL_PATH = "model/model.joblib"
TARGET_COL = "target"


def train(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, 
            random_state=42,
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5
        )
        ),
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Accuracy: {score:.4f}")
    return pipeline


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    data_path = "data/current_data.csv"
    df = pd.read_csv(data_path)
    pipeline = train(df)
    save_model(pipeline)
