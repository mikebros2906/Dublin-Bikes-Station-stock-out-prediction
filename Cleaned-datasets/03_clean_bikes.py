import pandas as pd
import os
import glob # Find files matching a pattern within system https://www.w3schools.com/python/ref_module_glob.asp

BASE         = r"D:\Griffith College Docs year 2\01 Dissertation\Datasets"
BIKES_FOLDER = os.path.join(BASE, "Dublin Bikes dataset",
                            "extract data from JCDecaux", "historical_data")
OUTPUT       = r"D:\Griffith College Docs year 2\01 Dissertation\Cleaned-datasets\bikes_cleaned.csv"

print("=" * 60)
print("SCRIPT 3: DUBLIN BIKES HISTORICAL DATA CLEANER")
print("=" * 60)

# Searches for all the CSV Files
csv_files = sorted(glob.glob(os.path.join(BIKES_FOLDER, "*.csv")))
print(f"\nFound {len(csv_files)} CSV files total")

# Exclude 2021 monthly files, covered by 2021_Q4.csv at better resolution as these were copied files of the same data.
EXCLUDE_FILES = {"2021_10.csv", "2021_11.csv", "2021_12.csv"}
csv_files = [f for f in csv_files if os.path.basename(f) not in EXCLUDE_FILES]
print(f"Files excluded (duplicate coverage at lower resolution): {sorted(EXCLUDE_FILES)}")
print(f"Files to process: {len(csv_files)}\n")

# = STATION-LEVEL FIXES ============================
# Station 507: ORIEL STREET TEST TERMINAL. Confirmed non-public test station. IDs above 200 in old format were non-standard. Has no match in 2024+ GBFS network. Remove entirely.
# 507 was a test terminal and has thus been removed.
REMOVE_STATION_IDS = {507}
print(f"Stations excluded entirely: {REMOVE_STATION_IDS}")
print("507 = ORIEL STREET TEST TERMINAL\n")

# Station 46: STRAND STREET GREAT
# Capacity shows two different values across files due to a station reconfiguration. Normalise all rows for this station to the correct
# capacity of 40 (the value used in the majority of records).
# Station 46 has no match in 2024+ network as it was decommissioned before 2024.
STATION_CAPACITY_CORRECTIONS = {
    46: 40   # STRAND STREET GREAT: correct capacity confirmed as 40
}
print(f"Station capacity corrections to apply: {STATION_CAPACITY_CORRECTIONS}")
print("  Station 46 (STRAND STREET GREAT): capacity normalised to 40\n")

# = COLUMN MAPPINGS ==============================

# Format A: 2019 quarterly + 2021 Q1-Q4 (spaces, mixed case status)
# Column name currently    : standardised column name used in all the dataframes
FORMAT_A = {
    "STATION ID"           : "station_id",
    "TIME"                 : "datetime",
    "NAME"                 : "name",
    "BIKE STANDS"          : "capacity",
    "AVAILABLE BIKE STANDS": "docks",
    "AVAILABLE BIKES"      : "bikes",
    "STATUS"               : "status",
    "LATITUDE"             : "lat",
    "LONGITUDE"            : "lon",
}

# Format B: 2022_01 to 2024_02 (underscores, uppercase status)
FORMAT_B = {
    "STATION ID"           : "station_id",
    "TIME"                 : "datetime",
    "NAME"                 : "name",
    "BIKE_STANDS"          : "capacity",
    "AVAILABLE_BIKE_STANDS": "docks",
    "AVAILABLE_BIKES"      : "bikes",
    "STATUS"               : "status",
    "LATITUDE"             : "lat",
    "LONGITUDE"            : "lon",
}

# Format C: 2024_05 onwards GBFS
FORMAT_C = {
    "last_reported"      : "datetime",
    "station_id"         : "station_id",
    "name"               : "name",
    "capacity"           : "capacity",
    "num_docks_available": "docks",
    "num_bikes_available": "bikes",
    "lat"                : "lat",
    "lon"                : "lon",
}

# Columns to drop and columns to keep
DROP_COLS = ["LAST UPDATED", "ADDRESS", "address", "short_name",
             "region_id", "system_id", "last_updated"]

KEEP_COLS = ["datetime", "station_id", "name",
             "bikes", "docks", "capacity", "lat", "lon", "status"]

#  PROCESS EACH FILE
all_dfs  = []
skipped  = []

# initialising counters for audit trail
total_raw_rows        = 0
removed_station_507   = 0
removed_docks_over    = 0
removed_bikes_over    = 0
corrected_capacity_46 = 0

# loop through every bikes csv files
for filepath in csv_files:
    filename = os.path.basename(filepath)

    # return file size and skip empty files
    size = os.path.getsize(filepath)
    if size < 500:
        print(f"  SKIP (empty file): {filename} = {size} bytes")
        skipped.append(filename)
        continue

    # loads file into memory, strips leading spaces and return the total length
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip()
        cols = set(df.columns)
        total_raw_rows += len(df)

        # Detect format 
        if "num_bikes_available" in cols:
            fmt = "C"

            # no status column for this format; status updated to open if all are true, else updated as closed
            df["status"] = (
                (df["is_installed"] == True) &
                (df["is_renting"]   == True) &
                (df["is_returning"] == True)
            ).map({True: "OPEN", False: "CLOSED"})
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                         errors="ignore")
            df = df.drop(columns=["is_installed","is_renting",
                                   "is_returning"], errors="ignore")
            df = df.rename(columns=FORMAT_C)

        # format detected: B, remove unwanted columns 
        elif "AVAILABLE_BIKES" in cols:
            fmt = "B"
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
            df = df.rename(columns=FORMAT_B)

        # format A detected, remove/rename columns
        elif "AVAILABLE BIKES" in cols:
            fmt = "A"
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
            df = df.rename(columns=FORMAT_A)
            if "status" in df.columns:
                df["status"] = df["status"].str.upper()

        else:
            print(f"  SKIP (unrecognised format): {filename}")
            print(f"       Columns: {list(df.columns)}")
            skipped.append(filename)
            continue

        # Keep only needed columns
        cols_present = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols_present]

        # checks if these columns are present within the dataframe; If absent, prints the missing columns names
        required = {"datetime", "station_id", "bikes", "docks"}
        if not required.issubset(set(cols_present)):
            missing = required - set(cols_present)
            print(f"  SKIP (missing columns {missing}): {filename}")
            skipped.append(filename)
            continue

        # Parse datetime format into python recognizable format and reports lost rows
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["datetime"])
        dropped_dt = before - len(df)

        # Convert numeric columns loops through each numeric columns and converts them into a number
        for col in ["bikes", "docks", "capacity", "lat", "lon", "station_id"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # FIX 1: Remove station 507 (test terminal) 
        before_507 = len(df)
        df = df[~df["station_id"].isin(REMOVE_STATION_IDS)]
        removed_station_507 += before_507 - len(df)

        # FIX 2 Normalise station 46 capacity
        for station_id, correct_capacity in STATION_CAPACITY_CORRECTIONS.items():
            mask = df["station_id"] == station_id
            n_corrected = mask.sum()
            if n_corrected > 0:
                df.loc[mask, "capacity"] = correct_capacity
                corrected_capacity_46 += n_corrected

        # Remove rows with negative values
        df = df[(df["bikes"] >= 0) & (df["docks"] >= 0)]

        # Keep only OPEN stations
        # CLOSED station zeros are not stock-outs: they are offline stations
        if "status" in df.columns:
            df = df[df["status"] == "OPEN"]

        # Remove impossible rows: OPEN station with both bikes=0 AND docks=0 is a sensor reporting error, not a real operational state
        df = df[~((df["bikes"] == 0) & (df["docks"] == 0))]

        # FIX 3: Remove rows where docks > capacity (sensor errors)
        before_docks = len(df)
        df = df[df["docks"] <= df["capacity"]]
        removed_docks_over += before_docks - len(df)

        # FIX 4: Remove rows where bikes > capacity (sensor errors)
        before_bikes = len(df)
        df = df[df["bikes"] <= df["capacity"]]
        removed_bikes_over += before_bikes - len(df)

        # Add the cleaned dataframe to the collectio list
        all_dfs.append(df)
        note = f" (dropped {dropped_dt} bad datetime rows)" if dropped_dt > 0 else ""
        print(f"  OK [{fmt}]: {filename} has {len(df):,} rows{note}")

    # exception from try/except block
    except Exception as e:
        print(f"  ERROR: {filename} {e}")
        skipped.append(filename)

# COMBINE ALL FILES
print(f"\nCombining {len(all_dfs)} files...")
df_all = pd.concat(all_dfs, ignore_index=True)

# Remove exact duplicate rows
before_dupes = len(df_all)
df_all = df_all.drop_duplicates(subset=["datetime", "station_id"])
dupes_removed = before_dupes - len(df_all)
print(f"Duplicate rows removed: {dupes_removed:,}")

# Filter to 2019-2026
df_all = df_all[
    (df_all["datetime"] >= "2019-01-01") &
    (df_all["datetime"] <= "2026-12-31")
]

# Sort chronologically
df_all = df_all.sort_values(["datetime", "station_id"]).reset_index(drop=True)

# Add helper columns for merging later
# date: date-only string for calendar merge
# hour: integer hour for weather merge
df_all["date"] = df_all["datetime"].dt.date.astype(str)
df_all["hour"] = df_all["datetime"].dt.hour

# FULL AUDIT REPORT
print("\n" + "=" * 60)
print("CLEANING AUDIT REPORT")
print("=" * 60)
print(f"Raw rows across all files          : {total_raw_rows:,}")
print(f"Rows removed station 507         : {removed_station_507:,}")
print(f"Rows corrected station 46 cap    : {corrected_capacity_46:,}")
print(f"Rows removed docks > capacity    : {removed_docks_over:,}")
print(f"Rows removed bikes > capacity    : {removed_bikes_over:,}")
print(f"Rows removed both zeros (sensor) : counted in per-file processing")
print(f"Duplicate rows removed             : {dupes_removed:,}")
print(f"Final rows after all cleaning      : {len(df_all):,}")

print("\n--- DATASET SUMMARY ---")
print(f"Unique stations  : {df_all['station_id'].nunique()}")
print(f"Date range       : {df_all['datetime'].min()} to {df_all['datetime'].max()}")
print(f"Columns          : {list(df_all.columns)}")
print(f"\nMissing values:\n{df_all.isnull().sum()}")
print(f"\nFirst 3 rows:\n{df_all.head(3)}")
print(f"\nLast 3 rows:\n{df_all.tail(3)}")

if skipped:
    print(f"\nSkipped files ({len(skipped)}): {skipped}")

# SAVE and EXPORT the cleaned bikes file into a parquet format
df_all.to_csv(OUTPUT, index=False)
print(f"\nSaved CSV    : {OUTPUT}")

parquet_output = OUTPUT.replace(".csv", ".parquet")
df_all.to_parquet(parquet_output, index=False)
print(f"Saved Parquet: {parquet_output}")

print("\nDone.")