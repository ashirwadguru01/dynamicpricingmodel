"""
STEP 2 — Preprocess Data
Run: python step2_preprocess.py
"""

import pandas as pd

def preprocess():
    df = pd.read_csv("retail_data.csv")

    # Convert season text → number
    season_map     = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
    df["season_num"] = df["season"].map(season_map)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Save cleaned file
    df.to_csv("retail_data_clean.csv", index=False)

    print("✅ retail_data_clean.csv saved")
    print(f"   Rows    : {len(df)}")
    print(f"   Columns : {list(df.columns)}")

if __name__ == "__main__":
    preprocess()
