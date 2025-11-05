import pandas as pd
import pytest
from pathlib import Path

from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.preprocess import (
    clean_and_merge, drop_duplicates, remove_outliers, preprocess_splits
)
from src.feature_pipeline.feature_engineering import (
    add_date_features, frequency_encode, target_encode, drop_unused_columns, run_feature_engineering
)
# ------------------------------------
# load.py – unit test
# ------------------------------------
# Confirms time-based splitting works.
def test_load_and_split_data(tmp_path) -> None:
    dummy_path = tmp_path / 'raw.csv' # create a dummy path in the temp directory
    # Create a dummy dataset
    df = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        "zipcode": [1000, 2000, 1000, 2000, 3000, 4000],
        "city_full": ["A", "B", "A", "B", "C", "D"],
    })
    df.to_csv(dummy_path, index=False) # save the dummy dataset to CSV

    # Run load and split
    train_df, eval_df, holdout_df = load_and_split_data(raw_path=str(dummy_path), output_dir=tmp_path)

    # Check splits
    assert not train_df.empty and not eval_df.empty and not holdout_df.empty # ensure none are empty
    assert train_df['date'].max() < pd.to_datetime('2020-01-01') # train before eval cutoff
    assert eval_df["date"].min() >= pd.to_datetime("2020-01-01") # eval after train cutoff
    assert holdout_df["date"].min() >= pd.to_datetime("2022-01-01") # holdout after eval cutoff
    assert (tmp_path / 'train.csv').exists() # check train file saved
    print("Load and split test passed.")

# ------------------------------------
# preprocess.py – unit tests
# ------------------------------------
# Confirms preprocessing functions behave as intended.
def test_remove_outliers_drop_high_prices() -> None:
    df = pd.DataFrame({
        'median_list_price': [100_000, 20_000_000, 30_000_000, 15_000_000, 500_000]
    })
    cleaned = remove_outliers(df)
    assert cleaned['median_list_price'].max() <= 19_000_000
    print("Outlier removal test passed.")

def testdrop_duplicates_remove_duplicates() -> None:
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"],
        "year": [2020, 2020],
        "median_list_price": [100, 100]
    })
    cleaned = drop_duplicates(df)

    assert cleaned.shape[0] == 0 # both rows are duplicates and should be dropped -> 0 rows remain
    # this is only true due to the nature of my test data, in real data we may have non-duplicate rows remaining
    print("Duplicate removal test passed.")

def test_clean_and_merge_skips_when_city_missing() -> None:
    df = pd.DataFrame({"date": ["2020-01-01"], "price": [100]})
    result = clean_and_merge(df, metros_path=None)  # should skip gracefully
    assert "date" in result.columns and "price" in result.columns
    print("Clean-and-merge (no city_full) passed")

# ------------------------------------
# feature_engineering – unit tests
# ------------------------------------
# Confirms feature functions create consistent numeric features and avoid leakage.
def test_add_date_features_extracts_parts() -> None:
    df = pd.DataFrame({"date": ["2020-01-15"]})
    df = add_date_features(df)
    assert df.loc[0, "year"] == 2020 and df.loc[0, "month"] == 1 and df.loc[0, "quarter"] == 1
    print("Date feature extraction test passed")

def test_frequency_encode_counts_values() -> None:
    train = pd.DataFrame({"zipcode": [1000, 1000, 2000]})
    eval = pd.DataFrame({"zipcode": [1000, 3000]})
    train, eval, freq_map = frequency_encode(train, eval, col="zipcode")

    assert train['zipcode_freq'].to_list() == [2,2,1] # 1000 appears twice in train, 2000 once
    assert eval['zipcode_freq'].to_list() == [2,0] # 1000 appears twice in train, 3000 not in train
    assert isinstance(freq_map, pd.Series)
    print("Frequency encoding test passed.")

def test_target_encode_applies_mapping() -> None:
    train = pd.DataFrame({"city_full": ["A", "B", "A"], "price": [100, 200, 300]})
    eval = pd.DataFrame({"city_full": ["A", "B"]})
    train, eval, te = target_encode(train, eval, "city_full", "price")
    assert "city_full_encoded" in train.columns
    assert eval["city_full_encoded"].notnull().all()
    # .notnull() outputs a Series of booleans, .all() checks all are True and returns a single boolean
    assert te is not None
    print("Target encoding test passed")

def test_drop_unused_columns_removes_leakage() -> None:
    train = pd.DataFrame({
        "date": ["2020-01-01"], "city_full": ["A"], "zipcode": [1000],
        "median_sale_price": [200], "price": [300]
    })
    eval = train.copy()
    train_dropped, eval_dropped = drop_unused_columns(train, eval)
    for col in ["date", "city_full", "zipcode", "median_sale_price"]:
        assert col not in train_dropped.columns and col not in eval_dropped.columns
    print("Leakage removal test passed")

# ------------------------------------
# feature engineering - integration test
# ------------------------------------
# Confirms the whole feature pipeline works together.
def test_full_pipeline_integration(tmp_path) -> None:
    raw = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=6, freq="365D"),
        "price": [100, 200, 300, 400, 500, 600],
        "zipcode": [1000, 2000, 1000, 2000, 1000, 2000],
        "city_full": ["A", "B", "A", "B", "A", "B"],
        "median_list_price": [150_000]*6
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    # Load and split
    train_df, eval_df, holdout_df = load_and_split_data(raw_path=str(raw_path), output_dir=tmp_path)
 
    processed_dir = tmp_path / "processed" # define processed directory path
    processed_dir.mkdir(parents=True, exist_ok=True) # ensure dir
    train_df.to_csv(tmp_path / "train.csv", index=False) # save train split
    eval_df.to_csv(tmp_path / "eval.csv", index=False) # save eval split
    holdout_df.to_csv(tmp_path / "holdout.csv", index=False) # save holdout split

    # preprocess splits
    preprocess_splits(
        splits='train',
        raw_dir=tmp_path,
        processed_dir=processed_dir,
        metros_path=None
    )
    preprocess_splits(
        splits='eval',
        raw_dir=tmp_path,
        processed_dir=processed_dir,
        metros_path=None
    )
    preprocess_splits(
        splits='holdout',
        raw_dir=tmp_path,
        processed_dir=processed_dir,
        metros_path=None
    )

    # feature engineering
    out_train, out_eval, out_holdout, freq_map, target_encoder = run_feature_engineering(
        in_train_path=processed_dir / "cleaning_train.csv",
        in_eval_path=processed_dir / "cleaning_eval.csv",
        in_holdout_path=processed_dir / "cleaning_holdout.csv",
        output_dir=processed_dir,
    )

    assert {'year', 'zipcode_freq', 'city_full_encoded'}.issubset(out_train.columns)
    assert {'year', 'zipcode_freq', 'city_full_encoded'}.issubset(out_eval.columns)
    assert freq_map is not None
    assert target_encoder is not None
    print("Full pipeline integration test passed.")



 