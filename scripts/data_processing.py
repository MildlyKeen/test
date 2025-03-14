import pandas as pd
import numpy as np

def optimize_memory(df):
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/heart_failure.csv")
    df = optimize_memory(df)
    df.to_csv("data/heart_failure_optimized.csv", index=False)
    print("✅ Memory optimization complete!")
