import pandas as pd
import os

# target folders/files
BASE            = r"D:\Griffith College Docs year 2\01 Dissertation\Datasets"
CALENDAR_FOLDER = os.path.join(BASE, "Irish calendar", "calendar csv")
OUTPUT          = r"D:\Griffith College Docs year 2\01 Dissertation\Cleaned-datasets\calendar_cleaned.csv"

print("=" * 50)
print("SCRIPT 2: IRISH CALENDAR CLEANER")
print("=" * 50)

# builts daily calendar from 2019 to 2026 and store them in dfs
print("\nLoading calendar files...")
cal_years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
dfs = []

# loops through each year and counts rows and stores file name in dfs
for year in cal_years:
    path = os.path.join(CALENDAR_FOLDER, f"{year}.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  {year}.csv loaded: {len(df)} rows")
    else:
        print(f"  WARNING: {year}.csv not found: skipping")

# Combine all years in one big data frame and removes the roginal row numbers to make them into continuation
df_cal = pd.concat(dfs, ignore_index=True)

# convert datetime column into python object and drop date columns with NaN values
df_cal["date"] = pd.to_datetime(df_cal["date"], errors="coerce")
df_cal = df_cal.dropna(subset=["date"])

# keep only public holidays, keeps only date columns
df_holidays = df_cal[df_cal["type"] == "Public holiday"][["date"]].copy()

# Removes duplicates
df_holidays = df_holidays.drop_duplicates(subset="date")

# assign every public holiday a value of 1 
df_holidays["is_public_holiday"] = 1
print(f"\nPublic holiday dates found: {len(df_holidays)}")

# building the cleaned csv file of calendar with day_of_week, is_weekend, is_public_holiday and is_working_day flags for the calendar
# Calendar date starts from monday (0)
# build full daily date range 2019-2026
print("\nBuilding full daily calendar 2019-2026...")
full_calendar = pd.DataFrame({
    "date": pd.date_range("2019-01-01", "2026-12-31", freq="D")
})

# day of week
full_calendar["day_of_week"] = full_calendar["date"].dt.dayofweek

# weekend flag
full_calendar["is_weekend"] = full_calendar["day_of_week"].isin([5, 6]).astype(int)

# merge public holidays
full_calendar = full_calendar.merge(df_holidays, on="date", how="left")
full_calendar["is_public_holiday"] = full_calendar["is_public_holiday"].fillna(0).astype(int)

# Working day flag: 1, if either weekend/public holiday
full_calendar["is_working_day"] = (
    (full_calendar["is_weekend"] == 0) &
    (full_calendar["is_public_holiday"] == 0)
).astype(int)

# Print out the full audit report of the dataframe
print("\n--- CLEANED DATA SUMMARY ---")
print(f"Total rows       : {len(full_calendar):,}")
print(f"Date range       : {full_calendar['date'].min().date()} to {full_calendar['date'].max().date()}")
print(f"Public holidays  : {full_calendar['is_public_holiday'].sum()}")
print(f"Weekend days     : {full_calendar['is_weekend'].sum()}")
print(f"Working days     : {full_calendar['is_working_day'].sum()}")
print(f"Missing values   :\n{full_calendar.isnull().sum()}")
print(f"\nFirst 5 rows:\n{full_calendar.head(5)}")

# give out the output onto cleaned_calendar.csv
full_calendar.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
print("Done.")