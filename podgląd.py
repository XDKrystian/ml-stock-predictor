import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_parquet("data/src/data/silver/KGH.WA_merged_daily.parquet")




groups = {
    "OHLCV (KGH)": [c for c in df.columns if c.endswith("__kgh.wa")],
    "Benchmarks": [c for c in df.columns if "__^" in c or "__uup__" in c or "__pln=x__" in c],
    "Macro": [c for c in df.columns if c.startswith("fred_")],
    "Fundamentals": [c for c in df.columns if c.startswith("fund_")],
}

for name, cols in groups.items():
    if not cols:
        continue
    plt.figure(figsize=(12, 3))
    sns.heatmap(df[cols].isna(), cbar=False)
    plt.title(f"NaNy: {name}")
    plt.tight_layout()
    plt.show()