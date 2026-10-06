import pandas as pd
import os

# target folders
BASE   = r"D:\Griffith College Docs year 2\01 Dissertation\Datasets"
INPUT  = os.path.join(BASE, "Met Eireann weather data", "phoenix_park_hourly_raw.csv")
OUTPUT = r"D:\Griffith College Docs year 2\01 Dissertation\Cleaned-datasets\weather_cleaned.csv"

print("=" * 50)
print("SCRIPT 1: WEATHER DATA CLEANER")
print("=" * 50)

# skips the first 15 lines as it just conatins Metadata like station name, location, units, etc
print("\nLoading raw file...")
df = pd.read_csv(
    INPUT,
    skiprows=15,
    on_bad_lines="skip",
    low_memory=False
)
print(f"Raw rows loaded : {len(df):,}")
print(f"Raw columns     : {list(df.columns)}")

# drop quality indicator columns
df = df.drop(columns=[c for c in df.columns if c.startswith("ind")], errors="ignore")

# Rename date to datetime
df = df.rename(columns={"date": "datetime"})

# Parse datetime
df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M", errors="coerce")

# frop unparseable rows
before = len(df)
df = df.dropna(subset=["datetime"])
dropped = before - len(df)
if dropped > 0:
    print(f"Dropped {dropped} unparseable rows")

# filter to 2019-2026
df = df[(df["datetime"] >= "2019-01-01") & (df["datetime"] <= "2026-12-31")]

# convert weather columns to numeric
weather_cols = ["rain", "temp", "wetb", "dewpt", "vappr", "rhum", "msl"]
for col in weather_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort chronologically
df = df.sort_values("datetime").reset_index(drop=True)

# Final column order
df = df[["datetime"] + weather_cols]

print("\n--- CLEANED DATA SUMMARY ---")
print(f"Rows         : {len(df):,}")
print(f"Date range   : {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
print("Done.")