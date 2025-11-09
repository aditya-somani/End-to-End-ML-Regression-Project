# Goal: Create a FastAPI app to serve our trained ML model into a web service that anyone 
# (or any system) can call over HTTP.

from fastapi import FastAPI            # Web framework for APIs
from pathlib import Path               # For handling file paths cleanly
from typing import Any                 # For type hints (clarity in endpoints)
import pandas as pd                    # To handle incoming JSON as DataFrames
import boto3                           # AWS SDK for Python
import os                              # env variables

# Import inference pipeline
from src.inference_pipeline.inference import TRAIN_FE_PATH, TRAIN_FEATURE_COLUMNS, predict

# Config
S3_BUCKET = os.getenv("S3_BUCKET","housing-regression-data-aditya-somani-ml-project")
REGION = os.getenv("REGION","us-east-1")
s3 = boto3.client("s3", region_name=REGION) # create S3 client

# Ensures your app always has the latest model/data locally, 
# but avoids re-downloading every time it starts.
def load_from_s3(
    key: str,
    local_path: str,
) -> Path:
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"Downloading {key} from S3 to {local_path}")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return local_path

# Paths
# Downloads model + training features from S3 if not cached.
MODEL_PATH = load_from_s3('models/xgb_best_model.pkl', 'models/xgb_best_model.pkl')
TRAIN_FE_PATH = load_from_s3('processed/feature_engineered_train.csv', 'data/processed/feature_engineered_train.csv')

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c!='price']
else:
    TRAIN_FEATURE_COLUMNS = None

# FastAPI app
# Instantiate the FastAPI app
app = FastAPI(title="Housing Regression API")

# / → simple landing endpoint to confirm API is alive.
@app.get('/')
def root() -> dict[str, str]:
    return {"message": "Housing Regression API is running"}

# /health → checks if model exists, returns status info (like expected feature count).
@app.get('/health')
def health() -> dict[str, Any]:
    status : dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status['status'] = "unhealthy"
        status['error'] = "Model not found"
    else:
        status['status'] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status['n_expected_features'] = len(TRAIN_FEATURE_COLUMNS)
    return status

# Prediction Endpoint: This is the core ML serving endpoint.
@app.post('/predict')
def predict_batch(data: list[dict]) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}

    df = pd.DataFrame(data) # Convert incoming JSON to DataFrame
    if df.empty:
        return {"error": "No data provided"}
    
    preds_df = predict(df, model_path=MODEL_PATH)

    response = {
        "predictions": preds_df['predicted_price'].astype(float).to_list()
    }
    if 'actual_price' in preds_df.columns:
        response['actuals'] = preds_df['actual_price'].astype(float).to_list()


    return response

# I thought of writing the code for batch prediction, but it is not necessary for now due to resources and time constraints.

"""
Execution Order / Module Flow

1. Imports (FastAPI, pandas, boto3, your inference function).
2. Config setup (env vars → bucket/region).
3. S3 utility (load_from_s3).
4. Download + load model/artifacts (MODEL_PATH, TRAIN_FE_PATH).
5. Infer schema (TRAIN_FEATURE_COLUMNS).
6. Create FastAPI app (app = FastAPI).
7. Declare endpoints (/, /health, /predict).
"""