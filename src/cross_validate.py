"""
Runs cross-validation to check how consistent the model performs
across different train/test splits.
"""

import os
import sys
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# Filepaths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed.csv")

TARGET_COL = "activity_intensity"
RANDOM_STATE = 42


def main() -> None:

    # Load the preprocessed dataset
    if not os.path.exists(PROCESSED_PATH):
        print("ERROR: processed.csv not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_PATH)

    # Split features and target
    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found in processed.csv.")
        sys.exit(1)

    features = df.drop(columns=[TARGET_COL])
    target = df[TARGET_COL].astype(str)

    # Identify numeric vs categorical columns
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    # Handle missing numeric values
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("pass", "passthrough")
    ])

    # Handle missing categories and encode them
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define the model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    # Combine preprocessing and model into one pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Set up a 5fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Run cross-validation (weighted F1 works well for multi-class)
    scores = cross_val_score(pipeline, features, target, cv=cv, scoring="f1_weighted")

    print("Cross-Validation Results (F1 weighted)")
    print(f"Scores: {scores}")
    print(f"Average: {scores.mean():.4f}")
    print(f"Std Dev: {scores.std():.4f}")

if __name__ == "__main__":
    main()