import pandas as pd
import re

# ==========================
# CONFIGURATION
# ==========================
INPUT_CSV = r"C:\Users\Ghofran MASSAOUDI\Desktop\HV_average_SetA.csv"   # your current file
OUTPUT_CSV = r"C:\Users\Ghofran MASSAOUDI\Desktop\HV_final_average_56.csv"

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(INPUT_CSV, sep=";")

# ==========================
# CLEAN INSTANCE NAMES
# ==========================
# Remove run identifiers and trailing parts so "c101_equitable_clean_Run01_ncro_results"
# becomes just "c101_equitable_clean"
df["Base_Instance"] = df["Instance"].apply(
    lambda x: re.sub(r"_Run\d+.*", "", str(x))
)

# ==========================
# COMPUTE AVERAGE ACROSS 10 RUNS
# ==========================
avg_df = (
    df.groupby("Base_Instance", as_index=False)
      .agg(Average_HV=("Average_HV", "mean"),
           Count_Runs=("Average_HV", "count"))
)

# ==========================
# SAVE RESULT
# ==========================
avg_df.to_csv(OUTPUT_CSV, sep=";", index=False)

# ==========================
# DISPLAY SUMMARY
# ==========================
print(f"✅ Final averages saved to: {OUTPUT_CSV}")
print(f"Total instances: {len(avg_df)}")
print(f"Example:\n{avg_df.head()}")
