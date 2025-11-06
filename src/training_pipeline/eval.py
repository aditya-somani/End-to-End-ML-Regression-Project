"""
Evaluate a saved XGBoost model on the eval split.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")

# Helper functions - will be used in testing as we won't be testing on whole data
def _maybe_sample(
    df: pd.DataFrame, 
    sample_frac: Optional[float], 
    random_state: int
) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Evaluate a saved XGBoost model on the eval split.

    Returns
    -------
    metrics : dict[str, float]
    """

    # load data
    eval_df = pd.read_csv(eval_path)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    # define target and features
    target = "price"
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    # load model
    model = load(model_path)
    y_pred = model.predict(X_eval)

    # calculate metrics
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("Evaluation:")
    print(f"\tMAE={mae:.2f} \n\tRMSE={rmse:.2f} \n\tR²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()