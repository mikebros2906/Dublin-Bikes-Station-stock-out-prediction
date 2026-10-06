import pandas as pd
import os

# PATHS
BASE          = r"D:\Griffith College Docs year 2\01 Dissertation"
CLEANED       = os.path.join(BASE, "Cleaned-datasets")
MERGED        = os.path.join(BASE, "merged-datasets")

BIKES_PATH    = os.path.join(CLEANED, "bikes_cleaned.parquet")
WEATHER_PATH  = os.path.join(CLEANED, "weather_cleaned.csv")
CALENDAR_PATH = os.path.join(CLEANED, "calendar_cleaned.csv")
OUTPUT_PATH   = os.path.join(MERGED,  "master_dataset.parquet")

os.makedirs(MERGED, exist_ok=True)

print("=" * 60)
print("SCRIPT 4: AGGREGATION, MERGE AND GAP IMPUTATION")
print("=" * 60)

# STEP 1: LOAD BIKES
print("\n[Step 1] Loading bikes data...")

# Read the cleaned bikes file (parquet) and print the audit report
df_bikes = pd.read_parquet(BIKES_PATH)
df_bikes["datetime"] = pd.to_datetime(df_bikes["datetime"])

print(f"  Loaded  : {len(df_bikes):,} rows")
print(f"  Stations: {df_bikes['station_id'].nunique()}")
print(f"  Range   : {df_bikes['datetime'].min()} to {df_bikes['datetime'].max()}")

# STEP 2: FLOOR DATETIME TO NEAREST HOUR
print("\n[Step 2] Flooring datetime to nearest hour...")

# Raw bike timestamps have seconds e.g. 00:00:03
# Weather timestamps are always exactly on the hour 00:00:00
# Flooring removes seconds so timestamps match for merging
df_bikes["datetime_hour"] = df_bikes["datetime"].dt.floor("h")

print(f"  Before : {df_bikes['datetime'].iloc[0]}")
print(f"  After  : {df_bikes['datetime_hour'].iloc[0]}")

# STEP 3: AGGREGATE BIKES FROM 5-MINUTE TO HOURLY
print("\n[Step 3] Aggregating bikes from 5-minute to hourly...")

# creating hourly columns which captures minute details within the hour 
# firstly, group all rows with same station AND same hour
df_hourly = df_bikes.groupby(["station_id", "datetime_hour"]).agg(
    bikes_min     = ("bikes",    "min"), # minimum bike count
    docks_min     = ("docks",    "min"), # minimum free docks
    bikes_mean    = ("bikes",    "mean"), # average bike count
    docks_mean    = ("docks",    "mean"), # average dock count (taking in changes in station size)
    name          = ("name",     "first"), # station name (does not change, so take the first)
    capacity      = ("capacity", "first"), # station capacity
    lat           = ("lat",      "first"), # latitude values for map-pointing
    lon           = ("lon",      "first"), # longitude values for map-pointing
    reading_count = ("bikes",    "count"), # counts reading in that hour
).reset_index()

# rounds the mean values to one decimal place
df_hourly["bikes_mean"] = df_hourly["bikes_mean"].round(1)
df_hourly["docks_mean"] = df_hourly["docks_mean"].round(1)

# prints the compression factor of how much compression was done; compressing 51 million rows to 7 million rows
print(f"  Before : {len(df_bikes):,} rows")
print(f"  After  : {len(df_hourly):,} rows")
print(f"  Factor : {len(df_bikes)/len(df_hourly):.1f}x reduction")

# the 51 million rows were later removed to free up memory
del df_bikes
print("  Freed bikes data from memory")

# STEP 4: LOAD AND MERGE WEATHER
print("\n[Step 4] Loading and merging weather data...")

# parse weather datetime into pandas object
df_weather = pd.read_csv(WEATHER_PATH)
df_weather["datetime_hour"] = pd.to_datetime(df_weather["datetime"])
df_weather = df_weather.drop(columns=["datetime"])
print(f"  Weather rows: {len(df_weather):,}")

# joins the weather columns onto the bikes dataset
df_merged = df_hourly.merge(df_weather, on="datetime_hour", how="left")

# calculates NaN for temp and prints report for matched/unmatched
weather_matched = df_merged["temp"].notna().sum()
weather_missing = df_merged["temp"].isna().sum()
print(f"  Matched    : {weather_matched:,} ({weather_matched/len(df_merged)*100:.2f}%)")
print(f"  Not matched: {weather_missing:,} ({weather_missing/len(df_merged)*100:.2f}%)")

# Keep weather loaded wihch will be needed again for imputation in later steps and delete bikes data as it has been merged and not in use right now
del df_hourly
print("  Freed hourly data from memory (weather kept for imputation)")

# STEP 5: LOAD AND MERGE CALENDAR
print("\n[Step 5] Loading and merging calendar data...")

# read the calendar files and strips the time value from datetime to match the calendar columns
df_calendar = pd.read_csv(CALENDAR_PATH)
df_merged["date"] = df_merged["datetime_hour"].dt.date.astype(str)

print(f"  Calendar rows: {len(df_calendar):,}")

# Joins the ccalendar columns (is_weekend, is_public_holiday and is_working_day) onto every station-hour
df_merged = df_merged.merge(df_calendar, on="date", how="left")

# prints matching percentage as audit report
calendar_matched = df_merged["is_public_holiday"].notna().sum()
print(f"  Matched: {calendar_matched:,} ({calendar_matched/len(df_merged)*100:.2f}%)")

# delete calendar data from memory
del df_calendar
print("  Freed calendar data from memory")

# STEP 6: ADD TIME-BASED FEATURES
# extract 4 time components and add them as different columns
print("\n[Step 6] Adding time-based features...")

df_merged["hour"]        = df_merged["datetime_hour"].dt.hour.astype("int8")
df_merged["month"]       = df_merged["datetime_hour"].dt.month.astype("int8")
df_merged["year"]        = df_merged["datetime_hour"].dt.year.astype("int16")
df_merged["day_of_week"] = df_merged["datetime_hour"].dt.dayofweek.astype("int8")

print("  Added: hour, month, year, day_of_week")

# STEP 7: FIX DTYPES
print("\n[Step 7] Fixing column dtypes for memory efficiency...")

df_merged["rhum"]              = df_merged["rhum"].astype("int16")
df_merged["station_id"]        = df_merged["station_id"].astype("int16")
df_merged["bikes_min"]         = df_merged["bikes_min"].astype("int8")
df_merged["docks_min"]         = df_merged["docks_min"].astype("int8")
df_merged["capacity"]          = df_merged["capacity"].astype("int8")
df_merged["reading_count"]     = df_merged["reading_count"].astype("int8")
df_merged["is_weekend"]        = df_merged["is_weekend"].astype("int8")
df_merged["is_public_holiday"] = df_merged["is_public_holiday"].astype("int8")
df_merged["is_working_day"]    = df_merged["is_working_day"].astype("int8")
df_merged["lat"]               = df_merged["lat"].astype("float32")
df_merged["lon"]               = df_merged["lon"].astype("float32")

# No gap flags needed yet, imputation fills the gaps
# near_gap will be set to 0 for all rows (no corrupted lag windows remain)
# imputed rows are marked with is_imputed for synthetic rows marking
df_merged["near_gap"]   = 0
df_merged["is_imputed"] = 0

print(f"  Memory after dtype fix: {df_merged.memory_usage(deep=True).sum()/1e9:.2f} GB")

# STEP 8: DEFINE BIKE AND WEATHER COLUMNS FOR IMPUTATION
BIKE_COLS    = ["station_id", "name", "lat", "lon", "capacity",
                "bikes_min", "docks_min", "bikes_mean", "docks_mean",
                "reading_count"]
WEATHER_COLS = ["rain", "temp", "wetb", "dewpt", "vappr", "rhum", "msl"]

# STEP 9: IMPUTE GAP 1: MARCH AND APRIL 2024
print("\n[Step 9] Imputing Gap 1: March and April 2024...")
print("  Source: March and April 2023 bike patterns")
print("  Weather: Real March and April 2024 weather data")

# Extract 2023 source bike data to be imputed on march and april 2024 gap data
source_gap1 = df_merged[
    (df_merged["datetime_hour"].dt.year  == 2023) &
    (df_merged["datetime_hour"].dt.month.isin([3, 4]))
][BIKE_COLS + ["datetime_hour"]].copy()

print(f"  Source rows from 2023: {len(source_gap1):,}")

# Shift to 2024: same month and day one year forward
# shifts date from 2023 to 2024 without altering anything else
source_gap1["datetime_hour"] = source_gap1["datetime_hour"] + pd.DateOffset(years=1)
print(f"  Shifted to: {source_gap1['datetime_hour'].min()} "
      f"to {source_gap1['datetime_hour'].max()}")

# Attach real 2024 weather for these dates
gap1_weather = df_weather[
    (df_weather["datetime_hour"].dt.year  == 2024) &
    (df_weather["datetime_hour"].dt.month.isin([3, 4]))
]
imputed_gap1 = source_gap1.merge(
    gap1_weather[["datetime_hour"] + WEATHER_COLS],
    on="datetime_hour", how="left"
)

# Add calendar features for actual 2024 dates
# changed date effects day of the year. Eg: 3/4/2023 was a Monday whereas it is Wednesday on the same date 2024
# recalculating is done to fix these
imputed_gap1["hour"]        = imputed_gap1["datetime_hour"].dt.hour.astype("int8")
imputed_gap1["month"]       = imputed_gap1["datetime_hour"].dt.month.astype("int8")
imputed_gap1["year"]        = imputed_gap1["datetime_hour"].dt.year.astype("int16")
imputed_gap1["day_of_week"] = imputed_gap1["datetime_hour"].dt.dayofweek.astype("int8")
imputed_gap1["date"]        = imputed_gap1["datetime_hour"].dt.date.astype(str)

# Easter Monday 2024 is the only public holiday in Mar-Apr 2024
# fixing the easter holiday problem where April 10, 2023 was easter but April 10, 2024 is not easter
easter_2024 = pd.Timestamp("2024-04-01").date()

# Checks if date is April 2024, True else False
imputed_gap1["is_public_holiday"] = (
    imputed_gap1["datetime_hour"].dt.date == easter_2024
).astype("int8")

# Flags saturdays and sundays correctly as they were not fixed when copying last year same date's data
imputed_gap1["is_weekend"] = (
    imputed_gap1["day_of_week"].isin([5, 6])
).astype("int8")

# Same, working day flags were fixed 
imputed_gap1["is_working_day"] = (
    (imputed_gap1["is_weekend"]        == 0) &
    (imputed_gap1["is_public_holiday"] == 0)
).astype("int8")

# All imputed gaps = 1 data flag added for synthetic data awareness making XGBoost to learn to create different weights for synthetic data
imputed_gap1["near_gap"]   = 0
imputed_gap1["is_imputed"] = 1

# Fix dtypes to match main dataset
imputed_gap1["rhum"]          = imputed_gap1["rhum"].astype("int16")
imputed_gap1["station_id"]    = imputed_gap1["station_id"].astype("int16")
imputed_gap1["bikes_min"]     = imputed_gap1["bikes_min"].astype("int8")
imputed_gap1["docks_min"]     = imputed_gap1["docks_min"].astype("int8")
imputed_gap1["capacity"]      = imputed_gap1["capacity"].astype("int8")
imputed_gap1["reading_count"] = imputed_gap1["reading_count"].astype("int8")
imputed_gap1["lat"]           = imputed_gap1["lat"].astype("float32")
imputed_gap1["lon"]           = imputed_gap1["lon"].astype("float32")

w_ok = imputed_gap1["temp"].notna().sum()
print(f"  Imputed rows created       : {len(imputed_gap1):,}")
print(f"  Rows with real 2024 weather: {w_ok:,}")

# STEP 10: IMPUTE GAP 2: SEPTEMBER 2024 DAYS 4-30
# same logic as for imputed gap 1
print("\n[Step 10] Imputing Gap 2: September 2024 days 4 to 30...")
print("  Source: September 2023 days 4 to 30 bike patterns")
print("  Weather: Real September 2024 weather data")

# copy data Sept, 2023 to Sept, 2024
source_gap2 = df_merged[
    (df_merged["datetime_hour"].dt.year  == 2023) &
    (df_merged["datetime_hour"].dt.month == 9) &
    (df_merged["datetime_hour"].dt.day   >  3)
][BIKE_COLS + ["datetime_hour"]].copy()
print(f"  Source rows from 2023: {len(source_gap2):,}")

# Shifts Sept, 2023 timestamps to Sept, 2024
source_gap2["datetime_hour"] = source_gap2["datetime_hour"] + pd.DateOffset(years=1)
print(f"  Shifted to: {source_gap2['datetime_hour'].min()} "
      f"to {source_gap2['datetime_hour'].max()}")

# Extracts sept, 2024 weather data and merge them 
gap2_weather = df_weather[
    (df_weather["datetime_hour"].dt.year  == 2024) &
    (df_weather["datetime_hour"].dt.month == 9) &
    (df_weather["datetime_hour"].dt.day   >  3)
]
imputed_gap2 = source_gap2.merge(
    gap2_weather[["datetime_hour"] + WEATHER_COLS],
    on="datetime_hour", how="left"
)

# calulcate actual day/weekend/public holidays
imputed_gap2["hour"]        = imputed_gap2["datetime_hour"].dt.hour.astype("int8")
imputed_gap2["month"]       = imputed_gap2["datetime_hour"].dt.month.astype("int8")
imputed_gap2["year"]        = imputed_gap2["datetime_hour"].dt.year.astype("int16")
imputed_gap2["day_of_week"] = imputed_gap2["datetime_hour"].dt.dayofweek.astype("int8")
imputed_gap2["date"]        = imputed_gap2["datetime_hour"].dt.date.astype(str)

# No public holidays in September 2024 days 4-30
# weekends marked and public holidays marked
imputed_gap2["is_public_holiday"] = 0
imputed_gap2["is_weekend"] = (
    imputed_gap2["day_of_week"].isin([5, 6])
).astype("int8")
imputed_gap2["is_working_day"] = (
    (imputed_gap2["is_weekend"]        == 0) &
    (imputed_gap2["is_public_holiday"] == 0)
).astype("int8")

imputed_gap2["near_gap"]   = 0
imputed_gap2["is_imputed"] = 1

# Fix dtypes
imputed_gap2["rhum"]          = imputed_gap2["rhum"].astype("int16")
imputed_gap2["station_id"]    = imputed_gap2["station_id"].astype("int16")
imputed_gap2["bikes_min"]     = imputed_gap2["bikes_min"].astype("int8")
imputed_gap2["docks_min"]     = imputed_gap2["docks_min"].astype("int8")
imputed_gap2["capacity"]      = imputed_gap2["capacity"].astype("int8")
imputed_gap2["reading_count"] = imputed_gap2["reading_count"].astype("int8")
imputed_gap2["lat"]           = imputed_gap2["lat"].astype("float32")
imputed_gap2["lon"]           = imputed_gap2["lon"].astype("float32")

w_ok2 = imputed_gap2["temp"].notna().sum()
print(f"  Imputed rows created       : {len(imputed_gap2):,}")
print(f"  Rows with real 2024 weather: {w_ok2:,}")

del df_weather
print("  Freed weather data from memory")

# STEP 11: COMBINE ORIGINAL AND IMPUTED ROWS
print("\n[Step 11] Combining original and imputed rows...")

# Defines exact column order for final output
FINAL_COLS = [
    "datetime_hour", "station_id", "name",
    "lat", "lon", "capacity",
    "bikes_min", "docks_min", "bikes_mean", "docks_mean", "reading_count",
    "rain", "temp", "wetb", "dewpt", "vappr", "rhum", "msl",
    "hour", "month", "year", "day_of_week", "date",
    "is_weekend", "is_public_holiday", "is_working_day",
    "near_gap", "is_imputed"
]

# Align columns across all three dataframes
df_merged      = df_merged[FINAL_COLS]
imputed_gap1   = imputed_gap1.reindex(columns=FINAL_COLS)
imputed_gap2   = imputed_gap2.reindex(columns=FINAL_COLS)

# combines all dataframes vertically
df_final = pd.concat([df_merged, imputed_gap1, imputed_gap2], ignore_index=True)

# memory cleared of junk
del df_merged, imputed_gap1, imputed_gap2
print("  Freed intermediate dataframes from memory")

# Sort combined data chronologically then by station
df_final = df_final.sort_values(
    ["datetime_hour", "station_id"]
).reset_index(drop=True)

# Remove any duplicates at boundaries
before_dedup = len(df_final)
df_final = df_final.drop_duplicates(subset=["datetime_hour", "station_id"])
dupes_removed = before_dedup - len(df_final)
if dupes_removed > 0:
    print(f"  Duplicate rows removed at boundaries: {dupes_removed:,}")

print(f"  Real rows          : {(df_final['is_imputed']==0).sum():,}")
print(f"  Imputed rows       : {(df_final['is_imputed']==1).sum():,}")
print(f"  Total rows         : {len(df_final):,}")

# STEP 12: VALIDATION REPORT
print("\n" + "=" * 60)
print("FINAL VALIDATION REPORT")
print("=" * 60)

print(f"\nTotal rows             : {len(df_final):,}")
print(f"Real rows              : {(df_final['is_imputed']==0).sum():,}")
print(f"Imputed rows           : {(df_final['is_imputed']==1).sum():,}")
print(f"Unique stations        : {df_final['station_id'].nunique()}")
print(f"Date range             : {df_final['datetime_hour'].min()} "
      f"to {df_final['datetime_hour'].max()}")
print(f"Memory usage           : {df_final.memory_usage(deep=True).sum()/1e9:.2f} GB")
print(f"Columns                : {len(df_final.columns)}")

print(f"\n--- MISSING VALUES ---")
missing = df_final.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) == 0:
    print("  None")
else:
    for col, count in missing_cols.items():
        print(f"  {col}: {count:,} ({count/len(df_final)*100:.2f}%)")

print(f"\n--- INTEGRITY CHECKS ---")

# counts rows with combi of timestamp and station id appear more than once
print(f"  Duplicate station-hours : "
      f"{df_final.duplicated(['datetime_hour','station_id']).sum()} (should be 0)")

# Counts impossible negative bike counts and impossible bikes > capacity
print(f"  Negative bikes_min      : {(df_final['bikes_min']<0).sum()} (should be 0)")
print(f"  Negative docks_min      : {(df_final['docks_min']<0).sum()} (should be 0)")
print(f"  bikes_min > capacity    : "
      f"{(df_final['bikes_min']>df_final['capacity']).sum()} (should be 0)")
print(f"  Calendar logic errors   : "
      f"{len(df_final[(df_final.is_working_day==1)&((df_final.is_weekend==1)|(df_final.is_public_holiday==1))])} (should be 0)")

# Calendar logic error checks and also checks for all imputed rows
print(f"\n--- 2024 COVERAGE AFTER IMPUTATION ---")
df2024 = df_final[df_final["datetime_hour"].dt.year == 2024]
monthly_2024 = df2024.groupby(
    df2024["datetime_hour"].dt.to_period("M")
).size()
for period, count in monthly_2024.items():
    imputed = df2024[
        (df2024["datetime_hour"].dt.to_period("M") == period) &
        (df2024["is_imputed"] == 1)
    ]
    tag = f" ({len(imputed):,} imputed)" if len(imputed) > 0 else " (all real)"
    print(f"  {period}: {count:,} rows{tag}")

print(f"\n--- EVENT RATES ---")

# calculates stock-out rates
stockout_rate = (df_final["bikes_min"]==0).mean()*100
fullness_rate = (df_final["docks_min"]==0).mean()*100
print(f"  Stock-out rate (bikes_min=0) : {stockout_rate:.2f}%")
print(f"  Fullness rate (docks_min=0)  : {fullness_rate:.2f}%")

print(f"\n--- PARTIAL MONTHS ---")

# calculates partial months/incomplete data
monthly = df_final.groupby(df_final["datetime_hour"].dt.to_period("M")).size()
low = monthly[monthly < 70000] # month with <700000 rows is incomplete
for period, count in low.items():
    sub = df_final[df_final["datetime_hour"].dt.to_period("M")==period]
    days = sub["datetime_hour"].dt.day.nunique()
    print(f"  {period}: {count:,} rows (~{days} days) = partial month")

# STEP 13: SAVE
# Final output saved as parquet
print(f"\n[Step 13] Saving final dataset...")
df_final.to_parquet(OUTPUT_PATH, index=False)
size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"  Saved : {OUTPUT_PATH}")
print(f"  Size  : {size_mb:.1f} MB")