"""
Prepares the raw fitness dataset by cleaning the data,
creating an activity intensity label, and saving a processed version for training.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Filepaths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw_data.xlsx")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed.csv")


TARGET_COL = "activity_intensity"

#Creates low/moderate/high intensity labels using steps, calories, and active minutes.
def build_intensity_label(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype=int)

    # Give higher scores to rows with higher activity values
    for col in ["Steps_Taken", "Calories_Burned", "Active_Minutes"]:
        q1, q2 = df[col].quantile([0.33, 0.66])
        score += (df[col] >= q1).astype(int)
        score += (df[col] >= q2).astype(int)

    # Turn total score into 3 buckets
    s1, s2 = score.quantile([0.33, 0.66])
    labels = pd.Series("moderate", index=df.index, dtype="object")
    labels[score <= s1] = "low"
    labels[score >= s2] = "high"
    return labels

def main() -> None:
    # Load raw data
    df = pd.read_excel(RAW_PATH).dropna(how="all")

    # Drop identifier columns as they don’t help the model
    df = df.drop(columns=["User_ID", "Full Name"], errors="ignore")

    # Convert important numeric columns to numbers (bad values become NaN)
    numeric_cols = [
        "Age", "Height (cm)", "Weight (kg)",
        "Steps_Taken", "Calories_Burned", "Hours_Slept",
        "Water_Intake (L)", "Active_Minutes",
        "Heart_Rate (bpm)", "Stress_Level (1-10)"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing numeric values with the median
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Fill missing categorical values
    for c in ["Gender", "Workout_Type", "Mood"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    # Create the target label
    df[TARGET_COL] = build_intensity_label(df)

    # Scale numeric features (B1 requirement)
    feature_numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()
    if feature_numeric_cols:
        df[feature_numeric_cols] = scaler.fit_transform(df[feature_numeric_cols])

    # Save the processed dataset
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Saved processed dataset to: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()