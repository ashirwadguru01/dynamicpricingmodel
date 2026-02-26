"""
STEP 3 — Train Models
Run: python step3_train_models.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["price", "competitor_price", "discount", "season_num", "holiday_flag"]
TARGET   = "quantity_sold"

def train():
    df = pd.read_csv("retail_data_clean.csv")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Linear Regression ──────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print("📊 Linear Regression")
    print(f"   MAE : {mean_absolute_error(y_test, lr_pred):.2f}")
    print(f"   R²  : {r2_score(y_test, lr_pred):.4f}")

    # ── Random Forest ───────────────────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("\n🌲 Random Forest")
    print(f"   MAE : {mean_absolute_error(y_test, rf_pred):.2f}")
    print(f"   R²  : {r2_score(y_test, rf_pred):.4f}")

    # ── Save both models ────────────────────────
    joblib.dump(lr, "model_linear.pkl")
    joblib.dump(rf, "model_forest.pkl")
    print("\n✅ model_linear.pkl saved")
    print("✅ model_forest.pkl saved")

if __name__ == "__main__":
    train()
