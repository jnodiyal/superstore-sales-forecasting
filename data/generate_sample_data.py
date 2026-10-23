"""Create SYNTHETIC Superstore-like data for testing the pipeline.

This is NOT the real Global Superstore dataset. Use it only to check the code runs.
Report results from the real dataset (data/superstore.csv), never from this file.
"""
import numpy as np
import pandas as pd


def generate(path="data/sample_superstore.csv", seed=42):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2011-01-01", "2014-12-01", freq="MS")
    rows = []
    for i, m in enumerate(months):
        monthly_target = 40000 + 700 * i + 12000 * np.sin(2 * np.pi * (m.month - 3) / 12)
        n_orders = int(rng.integers(60, 90))
        shares = rng.dirichlet(np.ones(n_orders))
        for s in shares:
            day = int(rng.integers(1, 28))
            rows.append({
                "Order ID": f"ORD-{len(rows):06d}",
                "Order Date": f"{day:02d}-{m.month:02d}-{m.year}",
                "Sales": round(float(monthly_target * s * rng.normal(1, 0.05)), 2),
                "Category": rng.choice(["Furniture", "Office Supplies", "Technology"]),
                "Region": rng.choice(["East", "West", "Central", "South"]),
            })
    df = pd.DataFrame(rows)
    # inject realistic quality problems so the checks have something to find
    dup = df.sample(15, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(8, random_state=2).index, "Sales"] = np.nan
    df.loc[df.sample(5, random_state=3).index, "Order Date"] = np.nan
    df.loc[df.sample(3, random_state=4).index, "Sales"] = -10.0
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    generate()
    print("Wrote data/sample_superstore.csv (synthetic)")
