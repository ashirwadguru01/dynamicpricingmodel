"""
STEP 1 — Generate Data
Run: python step1_generate_data.py
"""

import numpy as np
import pandas as pd

def generate_data():
    rng = np.random.default_rng(42)
    n   = 2000

    seasons    = rng.choice(["Spring", "Summer", "Autumn", "Winter"], n)
    season_map = {"Spring": 1.0, "Summer": 1.2, "Autumn": 0.9, "Winter": 0.8}
    season_num = np.array([season_map[s] for s in seasons])

    price            = rng.uniform(20, 200, n).round(2)
    competitor_price = (price * rng.uniform(0.8, 1.2, n)).round(2)
    discount         = rng.uniform(0, 0.30, n).round(3)
    holiday_flag     = rng.choice([0, 1], n, p=[0.9, 0.1])
    effective_price  = price * (1 - discount)

    quantity_sold = (
        500
        - 1.8 * effective_price
        + 0.6 * competitor_price
        + 300 * discount
        + 50  * holiday_flag
        + 40  * season_num
        + rng.normal(0, 20, n)
    ).clip(0).round().astype(int)

    df = pd.DataFrame({
        "product_id":       rng.choice([f"P{i:03d}" for i in range(1, 11)], n),
        "price":            price,
        "competitor_price": competitor_price,
        "discount":         discount,
        "season":           seasons,
        "holiday_flag":     holiday_flag,
        "quantity_sold":    quantity_sold,
    })

    df.to_csv("retail_data.csv", index=False)
    print(f"✅ retail_data.csv saved — {len(df)} rows")
    print(df.head())

if __name__ == "__main__":
    generate_data()
