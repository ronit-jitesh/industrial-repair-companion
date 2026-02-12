#!/usr/bin/env python3
"""
clean_data.py
Phase 2: Data Engineering & Pandas
Loads the raw repair_logs.csv, cleans missing values, standardizes formats,
engineers a "Symptom_String" feature, and exports the cleaned dataset.
"""

import pandas as pd
import numpy as np
import os

# ─── Configuration ───────────────────────────────────────────────────────────

INPUT_FILE = "data/repair_logs.csv"
OUTPUT_FILE = "data/repair_logs_cleaned.csv"

# ─── Load Data ───────────────────────────────────────────────────────────────


def load_data(path):
    """Load raw repair logs CSV."""
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    return df


# ─── Data Quality Report ────────────────────────────────────────────────────


def print_quality_report(df, title="Data Quality Report"):
    """Print comprehensive data quality statistics."""
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")

    # Missing values
    print("\n📉 Missing Values:")
    missing = df.isnull().sum()
    for col in df.columns:
        m = missing[col]
        if m > 0:
            pct = m / len(df) * 100
            print(f"   {col:<25s}: {m:>6,} ({pct:.1f}%)")

    # Data types
    print("\n🔢 Data Types:")
    for col in df.columns:
        print(f"   {col:<25s}: {df[col].dtype}")

    # Unique values for categorical columns
    cat_cols = ["Error_Code", "Outcome", "Site_Location", "Machine_ID"]
    print("\n📋 Unique Values (categorical):")
    for col in cat_cols:
        if col in df.columns:
            print(f"   {col:<25s}: {df[col].nunique():,}")

    # Numeric summaries
    num_cols = ["Operating_Temp", "Vibration_Level", "Humidity"]
    print("\n📐 Numeric Summaries:")
    for col in num_cols:
        if col in df.columns:
            s = df[col].describe()
            print(f"   {col}: min={s['min']:.1f}, mean={s['mean']:.1f}, max={s['max']:.1f}, std={s['std']:.1f}")


# ─── Cleaning Steps ─────────────────────────────────────────────────────────


def clean_data(df):
    """Apply all cleaning transformations."""

    print("\n🔧 Step 1: Standardizing text fields...")

    # Uppercase error codes and strip whitespace
    df["Error_Code"] = df["Error_Code"].str.strip().str.upper()
    df["Error_Description"] = df["Error_Description"].str.strip().str.title()

    # Clean technician notes — strip whitespace, normalize
    df["Technician_Notes"] = df["Technician_Notes"].str.strip()

    # Standardize site location names
    df["Site_Location"] = df["Site_Location"].str.strip().str.title()

    # Standardize outcome values
    df["Outcome"] = df["Outcome"].str.strip().str.title()

    print("   ✅ Text fields standardized")

    # ──────────────────────────────────────────────────────────────────────

    print("\n🔧 Step 2: Handling missing Operating_Temp values...")

    # Strategy: Fill with median temperature grouped by Site_Location
    # Rationale: Temperature varies by geographic location / plant conditions
    temp_before = df["Operating_Temp"].isnull().sum()

    site_medians = df.groupby("Site_Location")["Operating_Temp"].transform("median")
    df["Operating_Temp"] = df["Operating_Temp"].fillna(site_medians)

    # If any still missing (entire site has no data), use global median
    global_temp_median = df["Operating_Temp"].median()
    df["Operating_Temp"] = df["Operating_Temp"].fillna(global_temp_median)

    temp_after = df["Operating_Temp"].isnull().sum()
    print(f"   Filled {temp_before - temp_after} missing values (site-level median imputation)")
    print(f"   Remaining nulls: {temp_after}")

    # ──────────────────────────────────────────────────────────────────────

    print("\n🔧 Step 3: Handling missing Vibration_Level values...")

    # Strategy: Fill with median vibration grouped by Machine_ID
    # Rationale: Each machine has its own baseline vibration profile
    vib_before = df["Vibration_Level"].isnull().sum()

    machine_medians = df.groupby("Machine_ID")["Vibration_Level"].transform("median")
    df["Vibration_Level"] = df["Vibration_Level"].fillna(machine_medians)

    # Fallback: use error-code-level median (some errors cause higher vibration)
    error_medians = df.groupby("Error_Code")["Vibration_Level"].transform("median")
    df["Vibration_Level"] = df["Vibration_Level"].fillna(error_medians)

    # Final fallback: global median
    global_vib_median = df["Vibration_Level"].median()
    df["Vibration_Level"] = df["Vibration_Level"].fillna(global_vib_median)

    vib_after = df["Vibration_Level"].isnull().sum()
    print(f"   Filled {vib_before - vib_after} missing values (machine-level → error-level → global median)")
    print(f"   Remaining nulls: {vib_after}")

    # ──────────────────────────────────────────────────────────────────────

    print("\n🔧 Step 4: Handling missing Humidity values...")

    # Strategy: Forward-fill within Site_Location groups sorted by date
    # Rationale: Humidity at a site changes slowly, nearby dates have similar values
    hum_before = df["Humidity"].isnull().sum()

    df = df.sort_values(["Site_Location", "Date"])
    df["Humidity"] = df.groupby("Site_Location")["Humidity"].transform(
        lambda x: x.ffill().bfill()
    )

    # Final fallback: global median
    global_hum_median = df["Humidity"].median()
    df["Humidity"] = df["Humidity"].fillna(global_hum_median)

    hum_after = df["Humidity"].isnull().sum()
    print(f"   Filled {hum_before - hum_after} missing values (forward-fill within site)")
    print(f"   Remaining nulls: {hum_after}")

    # ──────────────────────────────────────────────────────────────────────

    print("\n🔧 Step 5: Parsing and standardizing dates...")

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day_of_Week"] = df["Date"].dt.day_name()

    print("   ✅ Added Year, Month, Day_of_Week columns")

    # ──────────────────────────────────────────────────────────────────────

    print("\n🔧 Step 6: Rounding numerical values...")

    df["Operating_Temp"] = df["Operating_Temp"].round(1)
    df["Vibration_Level"] = df["Vibration_Level"].round(1)
    df["Humidity"] = df["Humidity"].round(0).astype(int)

    print("   ✅ Temp to 1 decimal, Vibration to 1 decimal, Humidity to integer")

    # Re-sort by Log_ID for consistent ordering
    df = df.sort_values("Log_ID").reset_index(drop=True)

    return df


# ─── Feature Engineering ────────────────────────────────────────────────────


def engineer_features(df):
    """Create derived features for downstream RAG/ML use."""

    print("\n🧪 Feature Engineering...")

    # 1. Symptom String — combines error context for embedding
    df["Symptom_String"] = (
        df["Error_Code"] + " | " +
        df["Error_Description"] + " | " +
        df["Technician_Notes"] + " | " +
        "Temp:" + df["Operating_Temp"].astype(str) + "°C" + " | " +
        "Vibration:" + df["Vibration_Level"].astype(str) + " | " +
        "Humidity:" + df["Humidity"].astype(str) + "%"
    )
    print("   ✅ Created Symptom_String (Error + Description + Notes + Environment)")

    # 2. Temperature Category — for filtering and analysis
    df["Temp_Category"] = pd.cut(
        df["Operating_Temp"],
        bins=[0, 30, 45, 60, 100],
        labels=["Normal", "Elevated", "High", "Critical"]
    )
    print("   ✅ Created Temp_Category (Normal/Elevated/High/Critical)")

    # 3. Vibration Category
    df["Vibration_Category"] = pd.cut(
        df["Vibration_Level"],
        bins=[0, 2.5, 5.0, 7.5, 11],
        labels=["Low", "Moderate", "High", "Severe"]
    )
    print("   ✅ Created Vibration_Category (Low/Moderate/High/Severe)")

    # 4. Repair success flag
    df["Was_Fixed"] = (df["Outcome"] == "Fixed").astype(int)
    print("   ✅ Created Was_Fixed binary flag")

    return df


# ─── Export ──────────────────────────────────────────────────────────────────


def export_data(df, path):
    """Export cleaned dataframe to CSV."""
    df.to_csv(path, index=False)
    print(f"\n💾 Exported cleaned data → {path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)} ({list(df.columns)})")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    # Load
    df = load_data(INPUT_FILE)
    print_quality_report(df, "BEFORE Cleaning")

    # Clean
    df = clean_data(df)

    # Feature Engineer
    df = engineer_features(df)

    # Final report
    print_quality_report(df, "AFTER Cleaning")

    # Export
    export_data(df, OUTPUT_FILE)

    # Final summary
    print(f"\n{'='*70}")
    print("✅ Data Cleaning & Feature Engineering Complete!")
    print(f"{'='*70}")
    print(f"  📝 Preprocessed {len(df):,} repair logs")
    print(f"  🔧 Standardized error codes and technician notes")
    print(f"  🌡️  Imputed {df['Operating_Temp'].notna().sum():,} temperature values")
    print(f"  📊 Created Symptom_String for vector embedding")
    print(f"  📁 Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
