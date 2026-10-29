from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "retail_sales.csv")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
for col in ["region", "category", "customer_type"]:
    df[col] = df[col].astype("string").str.strip().str.title()
df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1).clip(lower=1)
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0)
df.loc[(df["discount_pct"] < 0) | (df["discount_pct"] > .8), "discount_pct"] = 0
df = df.drop_duplicates()
df["revenue"] = df["units"] * df["unit_price"] * (1-df["discount_pct"])
df["region"] = df["region"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")
df["month"] = df["order_date"].dt.to_period("M").astype(str)

df.describe(include="all").transpose().to_csv(out / "descriptive_statistics.csv")
df.groupby("region")["revenue"].agg(["count","sum","mean"]).sort_values("sum", ascending=False).to_csv(out / "region_summary.csv")
df.groupby("category")["revenue"].agg(["count","sum","mean"]).sort_values("sum", ascending=False).to_csv(out / "category_summary.csv")
df.groupby("month")["revenue"].sum().sort_index().to_csv(out / "monthly_revenue.csv")

top_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
top_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
monthly = df.groupby("month")["revenue"].sum().sort_index()
insights = [
    "EDA results are generated from synthetic practice data, not actual business records.",
    f"Highest-revenue category in this sample: {top_cat.index[0]} (${top_cat.iloc[0]:,.2f}).",
    f"Highest-revenue region in this sample: {top_region.index[0]} (${top_region.iloc[0]:,.2f}).",
    f"Total analyzed revenue: ${df['revenue'].sum():,.2f}.",
    f"Average order revenue: ${df['revenue'].mean():,.2f}.",
    "Use the monthly trend and category/region summaries to identify areas for further investigation; do not treat synthetic patterns as real-world conclusions."
]
(out / "eda_insights.txt").write_text("\n".join(insights))

plt.figure(figsize=(8,5)); top_cat.sort_values().plot(kind="barh"); plt.title("Total Revenue by Category"); plt.xlabel("Revenue"); plt.tight_layout(); plt.savefig(out/"category_revenue.png", dpi=150); plt.close()
plt.figure(figsize=(7,5)); top_region.sort_values().plot(kind="bar"); plt.title("Total Revenue by Region"); plt.ylabel("Revenue"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(out/"region_revenue.png", dpi=150); plt.close()
plt.figure(figsize=(9,5)); monthly.plot(marker="o"); plt.title("Monthly Revenue Trend"); plt.ylabel("Revenue"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(out/"monthly_trend.png", dpi=150); plt.close()
print("Task 3 complete. Outputs saved to:", out)
print("\n".join(insights))
