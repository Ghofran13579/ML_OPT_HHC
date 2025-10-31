import os
import re
import pandas as pd

# ==========================
# CONFIGURATION
# ==========================
ROOT_DIR = r"C:\Users\Ghofran MASSAOUDI\Desktop\Q-Learnung_NCRO_HV_Based-test"  # Folder containing run1...run10
OUTPUT_CSV = r"C:\Users\Ghofran MASSAOUDI\Desktop\HV_average_SetTest.csv"
VALID_EXTENSIONS = [".csv", ".txt"]
HV_PATTERN = re.compile(r'\bHV\s*[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)


# ==========================
# FUNCTION: Extract HV from file
# ==========================
def extract_hv_from_file(file_path):
    """Extract a single HV value from CSV or TXT file."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, dtype=str)
            hv_cols = [c for c in df.columns if "hv" in c.lower()]
            if hv_cols and not df.empty:
                hv_str = df[hv_cols[0]].dropna().iloc[0]

                parts = re.split(r'[;,]', hv_str)
                for part in reversed(parts):
                    try:
                        return float(part.strip())
                    except ValueError:
                        continue

        else:  # TXT
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                match = HV_PATTERN.search(text)
                if match:
                    return float(match.group(1))
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
    return None


# ==========================
# MAIN EXTRACTION
# ==========================
records = []

# Loop through each run folder (run1, run2, ..., run10)
run_folders = sorted([f for f in os.listdir(ROOT_DIR) if f.lower().startswith("run")])
if not run_folders:
    print(f"⚠️ No run folders found in {ROOT_DIR}")
else:
    print(f"🔍 Found {len(run_folders)} run folders: {run_folders}")

for run_folder in run_folders:
    run_path = os.path.join(ROOT_DIR, run_folder)
    if not os.path.isdir(run_path):
        continue

    for file in os.listdir(run_path):
        if any(file.lower().endswith(ext) for ext in VALID_EXTENSIONS):
            instance_name = os.path.splitext(file)[0]
            file_path = os.path.join(run_path, file)
            hv_value = extract_hv_from_file(file_path)
            if hv_value is not None:
                records.append({
                    "Run": run_folder,
                    "Instance": instance_name,
                    "HV": hv_value
                })


# ==========================
# COMPUTE AVERAGE PER INSTANCE
# ==========================
if not records:
    print("⚠️ No HV values found.")
else:
    df = pd.DataFrame(records)
    print(f"✅ Extracted {len(df)} HV values across {df['Run'].nunique()} runs.")

    # Compute mean HV per instance across runs
    avg_df = (
        df.groupby("Instance", as_index=False)
          .agg(Average_HV=("HV", "mean"), Count_Runs=("HV", "count"))
    )

    # Compute global average HV across all instances
    global_avg = avg_df["Average_HV"].mean()

    # Save to CSV
    avg_df.to_csv(OUTPUT_CSV, sep=";", index=False)

    # ==========================
    # DISPLAY SUMMARY
    # ==========================
    print(f"\n✅ Average HV per instance saved to: {OUTPUT_CSV}")
    print("=== SUMMARY ===")
    print(f"Total instances: {len(avg_df)}")
    print(f"Average HV across all instances (mean of means): {global_avg:.4f}")
