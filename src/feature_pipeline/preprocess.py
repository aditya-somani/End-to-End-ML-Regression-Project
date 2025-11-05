"""
Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}

def normalize_city(s: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    if pd.isna(s):
        return s
    
    s = str(s).strip().lower() # Lowercase + strip whitespace
    s = re.sub(r'[---]', '-', s) # unify dashes i.e. em-dash, en-dash, hyphen to hyphen
    s = re.sub(r'\s+', ' ', s) # unify spaces 
    return s

def clean_and_merge(
    df: pd.DataFrame,
    metros_path: str | None = 'data/raw/usmetros.csv',
) -> pd.DataFrame:
    """
    Normalize city names, optionally merge lat/lng from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """

    # make sure city_full exists
    if 'city_ful' not in df.columns:
        print("Skipping city merge: no 'city_full' column present.")
        return df
    
    # Normalize city full
    df['city_full'] = df["city_full"].apply(normalize_city)

    # apply mapping fixes
    norm_mapping = {normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()}
    df['city_full'] = df['city_full'].replace(norm_mapping)

    # If lat/lng already present, skip merge
    if {'lat', 'lng'}.subset(df.columns):
        print("Skipping lat/lng merge: already present in DataFrame.")
        return df
    
    # If no metros file provided / exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        print("Skipping lat/lng merge: no metros_path provided or file does not exist.")
        return df
    
    # Merge lat/lng from metros
    metros = pd.read_csv(metros_path)
    if 'metro_full' not in metros.columns or not {'lat', 'lng'}.subset(metros.columns):
        print("Skipping lat/lng merge: metros file missing required columns.")
        return df
    
    metros['metro_full'] = metros['metro_full'].apply(normalize_city)
    df = df.merge(
        metros[['metro_full', 'lat', 'lng']],
        how='left',
        left_on='city_full',
        right_on='metro_full',
    )
    df.drop(columns=['metro_full'], inplace=True, errors='ignore')

    # checking for any missing city lat/lng after merge
    missing = df[df['lat'].isna() | df['lng'].isna()]['city_full'].unique()
    if len(missing) > 0:
        print(f"Warning: Missing lat/lng for cities after merge: {missing}")
    else:
        print("All cities successfully merged with lat/lng.")
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""

    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(['date', 'year']), keep=False)
    after = df.shape[0]
    print(f"Dropped {before - after} duplicate rows(excluding date/year).")
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if 'median_list_price' not in df.columns:
        print("Skipping outlier removal: no 'median_list_price' column present.")
        return df
    before = df.shape[0]
    df = df[df['median_list_price'] <= 19_000_000]
    after = df.shape[0]
    print(f"Removed {before - after} outlier rows based on median_list_price.")
    return df

def preprocess_splits(
    splits: str, # train/eval/holdout => preprocess that split
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = 'data/raw/usmetros.csv',
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir) # ensure Path, if str then it will be converted to Path and if Path then it will remain Path 
    processed_dir = Path(processed_dir)  # ensure Path
    processed_dir.mkdir(parents=True, exist_ok=True) # ensure dir, if not exist then create it else do nothing

    path = raw_dir / f"{splits}.csv"  # construct path to the raw split file
    df = pd.read_csv(path) # read the raw split file into a DataFrame

    df = clean_and_merge(df, metros_path=metros_path) # clean and merge city names and lat/lng
    df = drop_duplicates(df) # drop duplicate rows
    df = remove_outliers(df) # remove extreme outliers

    out_path = processed_dir / f"cleaning_{splits}.csv" # construct path to save the processed split file
    df.to_csv(out_path, index=False) # save the processed DataFrame to CSV
    print(f"Saved cleaned {splits} split to {out_path}")

    return df

def run_preprocess(
    splits: tuple[str, ...] = ('train', 'eval', 'holdout'), # tuple of (train,eval,holdout) - default values
    # The ... means the tuple can have an arbitrary number of string elements, not just fixed-length.
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = 'data/raw/usmetros.csv',
) -> None:
    """Run preprocessing for all splits and save to processed_dir."""
    for s in splits:
        preprocess_splits(
            splits=s,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            metros_path=metros_path,
        )

if __name__=="__main__":
    run_preprocess()




