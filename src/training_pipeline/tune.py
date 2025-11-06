"""
Hyperparameter tuning with Optuna + MLflow.

- Optimizes XGB params on eval set RMSE.
- Logs trials to MLflow.
- Retrains best model and saves to `model_output`.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost

DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/xgb_best_model.pkl")

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

def _load_data(
    train_path: Path | str,
    eval_path: Path | str,
    sample_frac: Optional[float],
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # load data
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    # sample data
    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    # define target and features
    target = 'price'
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    return X_train, y_train, X_eval, y_eval

def tune_model(
    train_path: Path | str = DEFAULT_TRAIN, # Path to the training data
    eval_path: Path | str = DEFAULT_EVAL, # Path to the evaluation data
    model_output: Path | str = DEFAULT_OUT, # Path to the output model
    n_trials: int = 15, # Number of trials to run for hyperparameter tuning
    sample_frac: Optional[float] = None, # Fraction of the data to sample
    tracking_uri: Optional[str] = None, # Tracking URI for MLflow
    experiment_name: str = "xgboost_optuna_housing", # Name of the experiment
    random_state: int = 42 # Random state for reproducibility
) -> tuple[dict, dict]:
    """Run Optuna tuning; save best model; return (best_params, best_metrics)."""

    # check if tracking_uri is provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # load data and split
    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path, sample_frac, random_state)

    # define objective function
    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameters to tune
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        # Start a nested MLflow run for this trial to log parameters and metrics
        with mlflow.start_run(nested=True):
            # Instantiate and train the XGBoost regressor with the sampled hyperparameters
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            # Predict on evaluation set and calculate evaluation metrics
            y_pred = model.predict(X_eval)
            rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
            mae = float(mean_absolute_error(y_eval, y_pred))
            r2 = float(r2_score(y_eval, y_pred))

            # Log the parameters and metrics of this trial to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        return rmse # we are optimizing for RMSE, so we return it

    study = optuna.create_study(direction='minimize') # we are minimizing RMSE, so we use 'minimize'
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print(f"Best params from Optuna: {best_params}")

    # Retrain best model and save to model_output
    best_model = XGBRegressor(**{**best_params, "random_state": random_state, "n_jobs": -1, "tree_method": "hist"})
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_eval)
    # metrics
    best_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "mae": float(mean_squared_error(y_eval, y_pred)),
        "r2": float(r2_score(y_eval, y_pred)),
    }
    print(f"Best tuned model metrics: {best_metrics}")

    # save to models/
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)
    print(f'Best model save to {out}')

    # log best model to MLflow
    with mlflow.start_run(run_name="best_xgboost_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.xgboost.log_model(best_model, name="model")

    return best_params, best_metrics

if __name__ == "__main__":
    tune_model()
