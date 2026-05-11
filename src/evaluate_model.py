"""
Evaluates the trained model on a test set and prints
accuracy, precision, recall, and F1 score.
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Filepaths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed.csv")
PIPELINE_PATH = os.path.join(BASE_DIR, "artifacts", "pipeline.joblib")

TARGET_COL = "activity_intensity"
RANDOM_STATE = 42


def main() -> None:

    # Load the preprocessed dataset
    if not os.path.exists(PROCESSED_PATH):
        print("ERROR: processed.csv not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_PATH)

    # Load the trained model pipeline
    if not os.path.exists(PIPELINE_PATH):
        print("ERROR: pipeline.joblib not found. Run train_model.py first.")
        sys.exit(1)

    pipeline = joblib.load(PIPELINE_PATH)

    # Split features and target
    if TARGET_COL not in df.columns:
        print(f"ERROR: Target column '{TARGET_COL}' not found in processed.csv.")
        sys.exit(1)

    features = df.drop(columns=[TARGET_COL])
    target = df[TARGET_COL].astype(str)

    # Split the dataset into training and testing sets just like training
    features_train, features_test, target_train, target_test = train_test_split(
        features, target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE
    )

    # Predict on the test set
    predictions = pipeline.predict(features_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(target_test, predictions)
    precision = precision_score(target_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(target_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(target_test, predictions, average="weighted", zero_division=0)

    print("Model Evaluation Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    main()