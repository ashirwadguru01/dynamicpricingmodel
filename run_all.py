"""
RUN ALL STEPS IN ORDER
======================
Just run this one file to execute everything:

    python run_all.py
"""

import step1_generate_data
import step2_preprocess
import step3_train_models
import step4_simulate_pricing
import step5_visualize

print("=" * 45)
print("  Dynamic Pricing Intelligence System")
print("=" * 45)

print("\n━━━ STEP 1: Generate Data ━━━")
step1_generate_data.generate_data()

print("\n━━━ STEP 2: Preprocess Data ━━━")
step2_preprocess.preprocess()

print("\n━━━ STEP 3: Train Models ━━━")
step3_train_models.train()

print("\n━━━ STEP 4: Simulate Pricing ━━━")
import joblib, numpy as np, pandas as pd
model  = joblib.load("model_forest.pkl")
sim_df = step4_simulate_pricing.simulate(model, 90.0, 0.10, 1, 0)
best   = sim_df.loc[sim_df["revenue"].idxmax()]
sim_df.to_csv("simulation_results.csv", index=False)
print(f"✅ simulation_results.csv saved")
print(f"🏆 Best Price  : Rs{best['price']:.2f}")
print(f"   Revenue     : Rs{best['revenue']:,.2f}")

print("\n━━━ STEP 5: Visualize ━━━")
step5_visualize.plot()

print("\n" + "=" * 45)
print("  ✅ All done! Files created:")
print("     retail_data.csv")
print("     retail_data_clean.csv")
print("     model_linear.pkl")
print("     model_forest.pkl")
print("     simulation_results.csv")
print("     pricing_chart.png")
print("=" * 45)
