# Frontend for the Housing Regression MLE project

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3, os
from pathlib import Path

from src.api.main import S3_BUCKET

# Config
API_URL = os.environ.get('API_URL', "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv('S3_BUCKET', 'housing-regression-data-aditya-somani-ml-project')
REGION = os.getenv('AWS_REGION', 'us-east-1')

# Ensure API_URL ends with /predict
API_URL = API_URL.rstrip('/') # remove the trailing slash from the API_URL
if not API_URL.endswith('/predict'):
    API_URL += '/predict' # add the /predict suffix if it's not there

# Intialize the s3 client
s3 = boto3.client('s3', region_name=REGION)

def load_from_s3(
    key: str,
    local_path: str,
) -> str:
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"Downloading {key} from S3 to {local_path}")
        s3.download_file(S3_BUCKET, key, str(local_path))

    return str(local_path)

# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    'processed/feature_engineered_holdout.csv',
    'data/processed/feature_engineered_holdout.csv'
)
HOLDOUT_META_PATH = load_from_s3(
    'processed/cleaning_holdout.csv',
    'data/processed/cleaning_holdout.csv'
)

# Data loading

@st.cache_data # cache the data so that it doesn't need to be loaded every time the app is run
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=['date'])[['date', 'city_full']]

    if len(fe) != len(meta):
        st.warning("Feature engineered holdout and meta data have different lengths. Aligning by Index")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp['date'] = meta['date']
    disp['region'] = meta['city_full']
    disp['year'] = disp['date'].dt.year
    disp['month'] = disp['date'].dt.month
    disp['actual_price'] = fe['price']

    return fe, disp

fe_df, disp_df = load_data()

# UI
st.title("Housing Regression Prediction") # Title of the app

# Filtering Options (years, months, regions)
years = sorted(disp_df['year'].unique())
months = list(range(1, 13))
regions = ['All'] + sorted(disp_df['region'].unique())

# Filtering UI
col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox('Select Year', years, index=0)
with col2:
    month = st.selectbox('Select Month', months, index=0)
with col3:
    region = st.selectbox('Select Region', regions, index=0)

if st.button('Show Prediction'):
    mask = (disp_df['year'] == year) & (disp_df['month'] == month)
    if region != 'All':
        mask &= (disp_df['region'] == region)

    idx = disp_df.index[mask] # get the indices of the rows that match the mask

    if len(idx) == 0:
        st.warning("No data found for the selected filters")
    else:
        st.write(f'Running prediction for **{year}-{month:02d}** in | Region: **{region}**') # 02d - format the month as a 2 digit number

        payload = fe_df.loc[idx].to_dict(orient='records') # convert the dataframe to a list of dictionaries
        # orient='records' argument tells pandas: "Treat each row as a separate record (dictionary)."

        try:
            response = requests.post(API_URL, json=payload, timeout=60) # send the request to the API
            response.raise_for_status() # raise an exception if the request is not successful
            output = response.json() # get the response from the API in JSON format

            # Parse Output
            preds = output.get('predictions', [])
            actuals = output.get('actuals', None)

            # Build View
            view = disp_df.loc[idx, ['date', 'region', 'actual_price']].copy() # get the date, region, and actual price columns
            view = view.sort_values(by='date') # sort the dataframe by date
            view['prediction'] = pd.Series(preds, index=view.index).astype(float) # add the predictions column

            if actuals is not None and len(actuals) == len(view):
                view['actual_price'] = pd.Series(actuals, index=view.index).astype(float) # add the actual price column

            # Metrics
            mae = (view['prediction'] - view['actual_price']).abs().mean()
            rmse = ((view['prediction'] - view['actual_price'])**2).mean() ** 0.5
            avg_percent_error = ((view['prediction'] - view['actual_price']).abs() / view['actual_price']).mean() * 100 # Avergae Absolute Percentage Error(AAPE)
            # This metric is commonly used to gauge the accuracy of a forecasting or prediction model.

            st.subheader('Prediction v/s Actuals')
            st.dataframe(
                view[['date', 'region', 'actual_price', 'prediction']].reset_index(drop=True),
                use_container_width=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric('MAE', f'{mae:.2f}')
            with c2:
                st.metric('RMSE', f'{rmse:.2f}')
            with c3:
                st.metric("Average % Error", f'{avg_percent_error:.2f}%')

            # Yearly Trend Chart
            if region == 'All':
                year_data = disp_df[disp_df['year'] == year].copy() # get the data for the selected year
                idx_all = year_data.index # get the indices of the rows that match the mask
                payload_all = fe_df.loc[idx_all].to_dict(orient='records') # convert the dataframe to a list of dictionaries
                
                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get('predictions', [])

                year_data['predictions'] = pd.Series(preds_all, index=year_data.index).astype(float)

            else:
                year_data = disp_df[(disp_df['year'] == year) & (disp_df['region'] == region)].copy()
                idx_region = year_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient='records')

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()  
                preds_region = resp_region.json().get('predictions', [])

                year_data['predictions'] = pd.Series(preds_region, index=year_data.index).astype(float)

            # Aggregate by Month
            monthly_avg = year_data.groupby('month')[['actual_price', 'predictions']].mean().reset_index()

            # Highlisght selected month
            monthly_avg['highlight'] = monthly_avg['month'].apply(lambda m:'Selected'if m == month else 'Other')

            # Plot
            fig = px.line(
                data_frame=monthly_avg,
                x='month',
                y=['actual_price', 'predictions'],
                markers=True,
                labels={'value': 'Price', 'month': 'Month'},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}",
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor='red',
                opacity=0.1,
                layer='below',
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f'API call failed: {e}')
            st.exception(e)

else:
    st.info('Choose filters and click **Show Prediction** to compute')