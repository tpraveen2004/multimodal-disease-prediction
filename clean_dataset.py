import os
import pandas as pd
import numpy as np

IMAGE_DIR = "../data/covid_xray/images"
META_PATH = "../data/covid_xray/metadata.csv"
OUT_PATH = "../data/covid_xray/metadata_clean.csv"

valid_ext = (".png", ".jpg", ".jpeg")

df = pd.read_csv(META_PATH)
df = df.replace({np.nan: ""})

clean_rows = []

for idx, row in df.iterrows():
    filename = str(row.get("filename", "")).strip()

    if len(filename) == 0:
        continue

    if not filename.lower().endswith(valid_ext):
        continue

    full_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(full_path):
        continue

    # Valid row
    clean_rows.append(row)

clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(OUT_PATH, index=False)

print("Cleaning complete.")
print("Original rows:", len(df))
print("Valid rows:", len(clean_df))
