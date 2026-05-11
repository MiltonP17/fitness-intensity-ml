"""
Builds a supervised machine learning model to predict
activity intensity (low/moderate/high) from the preprocessed fitness dataset.
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

#Filepaths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR,"data", "processed.csv")
PIPELINE_PATH = os.path.join(BASE_DIR,"artifacts", "pipeline.joblib")

TARGET_COL = "activity_intensity"
RANDOM_STATE = 42


def main() -> None:

    # Load the preprocessed dataset
    df = pd.read_csv(PROCESSED_PATH)

    # Split features and target
    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found in processed.csv.")
        sys.exit(1)

    features = df.drop(columns=[TARGET_COL])
    target = df[TARGET_COL].astype(str)

    # Identify numeric vs categorical columns
    # Numeric columns are already scaled in preprocess.py, so we pass them through.
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    # Encode categorical features for model training
    # Numeric columns are already scaled in preprocess.py, but we still impute just in case.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("pass", "passthrough")
    ])

    # Fill missing categories and encode them for the model
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

    # Define a classification model suitable for structured fitness data
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    # Combine preprocessing steps and the model into a single pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Split the dataset into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(
        features, target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE
    )

    # Train the model using the training data
    pipeline.fit(features_train, target_train)

    # Save the trained pipeline for later evaluation and tuning
    joblib.dump(pipeline, PIPELINE_PATH)

    print(f"Model pipeline trained and saved to: {PIPELINE_PATH}")
    print("Model training complete. Ready for evaluation.")

if __name__ == "__main__":
    main()