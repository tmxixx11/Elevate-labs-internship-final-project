from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parent / "data" / "customer_churn.csv"
df = pd.read_csv(path)
print("Rows:", len(df))
print("Overall churn rate:", f"{df['churn'].eq('Yes').mean():.1%}")
print("\nChurn rate by contract:")
print((df.groupby("contract")["churn"].apply(lambda s: s.eq("Yes").mean()) * 100).round(2))
print("\nAverage monthly charges by churn status:")
print(df.groupby("churn")["monthly_charges"].mean().round(2))
print("\nMissing values:")
print(df.isna().sum())
