import pandas as pd
from pathlib import Path

# Input CSV
in_file = "analysis/ranking_similarity.csv"
# Output folder
out_dir = Path("analysis")
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "filtered.csv"

# Load
df = pd.read_csv(in_file)

# Condition: config contains 'en-de'
mask_en_de = df["config1"].str.contains("en-de") | df["config2"].str.contains("en-de")

# Keep only if lang is en or de OR config does not contain en-de
df_filtered = df[~mask_en_de | df["lang"].isin(["en", "de"])]

# Save
df_filtered.to_csv(out_file, index=False)
print(f"Filtered file saved to {out_file}")
