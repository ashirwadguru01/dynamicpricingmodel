"""
STEP 5 — Visualize Demand & Revenue Curves
Run: python step5_visualize.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def plot():
    sim_df = pd.read_csv("simulation_results.csv")
    best   = sim_df.loc[sim_df["revenue"].idxmax()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dynamic Pricing — Demand & Revenue Curves", fontsize=14, fontweight="bold")

    # ── Chart 1: Demand vs Price ──────────────────
    ax1.plot(sim_df["price"], sim_df["quantity"], color="#3b82f6", linewidth=2)
    ax1.axvline(best["price"], color="#ef4444", linestyle="--",
                label=f"Best price = ${best['price']:.2f}")
    ax1.set_title("Demand vs Price")
    ax1.set_xlabel("Price ($)")
    ax1.set_ylabel("Units Sold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Chart 2: Revenue vs Price ─────────────────
    ax2.plot(sim_df["price"], sim_df["revenue"], color="#22c55e", linewidth=2)
    ax2.axvline(best["price"], color="#ef4444", linestyle="--",
                label=f"Best price = ${best['price']:.2f}")
    ax2.scatter([best["price"]], [best["revenue"]], color="#ef4444", s=100, zorder=5,
                label=f"Max revenue = ${best['revenue']:,.0f}")
    ax2.set_title("Revenue vs Price")
    ax2.set_xlabel("Price ($)")
    ax2.set_ylabel("Revenue ($)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("pricing_chart.png", dpi=150, bbox_inches="tight")
    print("✅ pricing_chart.png saved")

if __name__ == "__main__":
    plot()
