"""
STEP 4 — Simulate Prices & Find Best Price
Run: python step4_simulate_pricing.py
"""

import joblib
import numpy as np
import pandas as pd

def simulate(model, competitor_price, discount, season_num, holiday_flag):
    prices  = np.linspace(10, 300, 500)
    results = []

    for p in prices:
        row = pd.DataFrame([{
            "price":            p,
            "competitor_price": competitor_price,
            "discount":         discount,
            "season_num":       season_num,
            "holiday_flag":     holiday_flag,
        }])
        qty     = max(0, float(model.predict(row)[0]))
        revenue = p * qty
        results.append({"price": round(p, 2), "quantity": round(qty, 1), "revenue": round(revenue, 2)})

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load the best model (Random Forest)
    model = joblib.load("model_forest.pkl")

    # --- Change these values to try different scenarios ---
    COMPETITOR_PRICE = 90.0
    DISCOUNT         = 0.10    # 10%
    SEASON           = 1       # 0=Spring 1=Summer 2=Autumn 3=Winter
    HOLIDAY          = 0       # 1=Yes 0=No
    # ------------------------------------------------------

    sim_df = simulate(model, COMPETITOR_PRICE, DISCOUNT, SEASON, HOLIDAY)
    best   = sim_df.loc[sim_df["revenue"].idxmax()]

    sim_df.to_csv("simulation_results.csv", index=False)

    print("✅ simulation_results.csv saved")
    print(f"\n🏆 Best Price      : ${best['price']:.2f}")
    print(f"   Units Sold     : {best['quantity']:.0f}")
    print(f"   Revenue        : ${best['revenue']:,.2f}")
