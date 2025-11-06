"""
Train a baseline XGBoost model.

- Reads feature-engineered train/eval CSVs.
- Trains XGBRegressor.
- Returns metrics and saves model to `model_output`.
"""

# from __future__ import annotations -> Not required as I am using python 3.11, which is >3.9
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DEFAULT_TRAIN = Path('data/processed/feature_engineered_train.csv')
DEFAULT_EVAL = Path('data/processed/feature_engineered_eval.csv')
DEFAULT_OUT = Path('models/xgb_model.pkl')

# Helper functions - will be used in testing as we won't be testing on whole data
def _maybe_sample(
    df: pd.DataFrame,
    sample_frac: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Sample dataframe if sample_frac is provided.

    Returns
    -------
    df : pd.DataFrame
    """
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac >= 1.0 or sample_frac <= 0.0:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def train_model(
    train_path: Path | str = DEFAULT_TRAIN, # Path to the training data
    eval_path: Path | str = DEFAULT_EVAL, # Path to the evaluation data
    model_output: Path | str = DEFAULT_OUT, # Path to the output model
    model_params: Optional[dict] = None, # Parameters for the model
    sample_frac: Optional[float] = None, # Fraction of the data to sample
    random_state: int = 42 # Random state for reproducibility
) -> tuple[XGBRegressor, dict]:
    """Train baseline XGB and save model.

    Returns
    -------
    model : XGBRegressor
    metrics : dict[str, float]
    """
    # load data
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    # sample data. Why? Because we want to avoid overfitting and to make the training process faster as this is 
    # basline model only.
    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    # define target and features
    target = 'price'
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    # params
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    # if model_params is provided, update the params
    if model_params:
        params.update(model_params)

    # train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval) # predict on evaluation set

    # calculate metrics
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    # save model
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True) # create directory if it doesn't exist
    dump(model, out) # save model
    print(f"Model trained and saved to {out}")
    print(f"Metrics: \n\tMAE={mae:.2f} \n\tRMSE={rmse:.2f} \n\tR²={r2:.4f}")

    return model, metrics

if __name__=='__main__':
    train_model()
