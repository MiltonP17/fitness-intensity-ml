"""
Tunes model hyperparameters to find a better performing setup.
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# Filepaths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed.csv")
BEST_PIPELINE_PATH = os.path.join(BASE_DIR, "artifacts", "best_pipeline.joblib")

TARGET_COL = "activity_intensity"
RANDOM_STATE = 42


def main() -> None:
    os.makedirs(os.path.join(BASE_DIR, "artifacts"), exist_ok=True)

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

    # Handle missing numeric values just in case
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("pass", "passthrough")]
    )

    # Handle missing categories and encode them
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Base model
    model = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Search space
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    # Cross validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring="f1_weighted",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(features, target)

    best_model = search.best_estimator_
    joblib.dump(best_model, BEST_PIPELINE_PATH)

    print("Hyperparameter Tuning Results")
    print(f"Best F1 (weighted): {search.best_score_:.4f}")
    print("Best parameters:")
    print(search.best_params_)
    print(f"\nSaved best pipeline to: {BEST_PIPELINE_PATH}")


if __name__ == "__main__":
    main()
