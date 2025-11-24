"""
Finance ERP API Service
Integrates data pipeline, cleaning, and forecasting models
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet and ARIMA imports
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize FastAPI app
app = FastAPI(
    title="Finance ERP API",
    description="API service for financial data pipeline, cleaning, and forecasting",
    version="1.0.0"
)

# Global variables to store models and data
current_data = None
prophet_model = None
arima_model = None
model_metrics = {}

# ==================== REQUEST/RESPONSE MODELS ====================

class DataInfo(BaseModel):
    rows: int
    columns: int
    column_names: List[str]
    missing_values: Dict[str, int]
    numeric_columns: List[str]
    date_columns: List[str]

class CleaningConfig(BaseModel):
    handle_missing: str = Field(default="ffill", description="Method: 'ffill', 'bfill', 'drop', 'mean'")
    handle_outliers: bool = Field(default=True, description="Whether to handle outliers using IQR method")
    normalize_columns: bool = Field(default=False, description="Whether to normalize column names")
    convert_dates: bool = Field(default=True, description="Whether to convert date columns")

class ForecastRequest(BaseModel):
    target_column: str = Field(description="Column to forecast (e.g., 'sales')")
    date_column: str = Field(description="Date column name")
    periods: int = Field(default=12, description="Number of periods to forecast")
    model_type: str = Field(default="both", description="Model to use: 'prophet', 'arima', or 'both'")
    train_test_split: float = Field(default=0.8, description="Train/test split ratio")

class ForecastResponse(BaseModel):
    model_used: str
    forecast_periods: int
    predictions: List[Dict[str, Any]]
    metrics: Dict[str, float]
    future_forecast: List[Dict[str, Any]]

# ==================== UTILITY FUNCTIONS ====================

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Detect potential date columns in dataframe"""
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
        elif df[col].dtype == 'object':
            # Try to parse as date
            try:
                pd.to_datetime(df[col].head(), errors='coerce')
                if df[col].head().notna().sum() > 0:
                    date_cols.append(col)
            except:
                pass
    return date_cols

def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Handle missing values using specified method"""
    if method == "ffill":
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == "bfill":
        return df.fillna(method='bfill').fillna(method='ffill')
    elif method == "drop":
        return df.dropna()
    elif method == "mean":
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        return df.fillna(method='ffill')
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers using IQR method"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_clean = df.copy()
    
    for col in numeric_cols:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_clean[col] = df_clean[col].clip(lower, upper)
    
    return df_clean

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, lowercase, replace spaces"""
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date columns to datetime"""
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def check_stationarity(timeseries) -> Dict[str, Any]:
    """Check stationarity using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return {
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'is_stationary': result[1] <= 0.05,
        'critical_values': {k: float(v) for k, v in result[4].items()}
    }

def calculate_metrics(actual, predicted) -> Dict[str, float]:
    """Calculate forecast evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2': float(r2)
    }

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Finance ERP API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload - Upload financial data (CSV/Excel)",
            "info": "/data/info - Get dataset information",
            "clean": "/data/clean - Clean dataset with specified config",
            "forecast": "/forecast - Generate forecasts using Prophet/ARIMA",
            "download": "/data/download - Download current cleaned dataset",
            "models": "/models/info - Get information about trained models"
        }
    }

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload financial data file (CSV or Excel)
    Returns basic information about the uploaded dataset
    """
    global current_data
    
    try:
        contents = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            current_data = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            current_data = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Get basic info
        info = {
            "filename": file.filename,
            "rows": len(current_data),
            "columns": len(current_data.columns),
            "column_names": current_data.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in current_data.dtypes.items()},
            "missing_values": current_data.isnull().sum().to_dict(),
            "sample_data": current_data.head(5).to_dict(orient='records')
        }
        
        return JSONResponse(content={
            "message": "File uploaded successfully",
            "data_info": info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/data/info")
def get_data_info():
    """Get detailed information about current dataset"""
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload data first.")
    
    numeric_cols = current_data.select_dtypes(include=np.number).columns.tolist()
    date_cols = detect_date_columns(current_data)
    
    info = DataInfo(
        rows=len(current_data),
        columns=len(current_data.columns),
        column_names=current_data.columns.tolist(),
        missing_values=current_data.isnull().sum().to_dict(),
        numeric_columns=numeric_cols,
        date_columns=date_cols
    )
    
    return {
        "data_info": info.dict(),
        "statistics": current_data.describe().to_dict(),
        "data_types": {col: str(dtype) for col, dtype in current_data.dtypes.items()}
    }

@app.post("/data/clean")
def clean_data(config: CleaningConfig):
    """
    Clean the current dataset using specified configuration
    Applies missing value handling, outlier detection, normalization
    """
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload data first.")
    
    try:
        df_cleaned = current_data.copy()
        operations_applied = []
        
        # Handle missing values
        df_cleaned = handle_missing_values(df_cleaned, config.handle_missing)
        operations_applied.append(f"Missing values handled using '{config.handle_missing}' method")
        
        # Handle outliers
        if config.handle_outliers:
            df_cleaned = handle_outliers(df_cleaned)
            operations_applied.append("Outliers handled using IQR method")
        
        # Normalize column names
        if config.normalize_columns:
            df_cleaned = normalize_column_names(df_cleaned)
            operations_applied.append("Column names normalized")
        
        # Convert date columns
        if config.convert_dates:
            df_cleaned = convert_date_columns(df_cleaned)
            operations_applied.append("Date columns converted")
        
        # Update current data
        current_data = df_cleaned
        
        return {
            "message": "Data cleaned successfully",
            "operations_applied": operations_applied,
            "cleaned_rows": len(df_cleaned),
            "cleaned_columns": len(df_cleaned.columns),
            "remaining_missing": df_cleaned.isnull().sum().sum()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning data: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    """
    Generate forecasts using Prophet and/or ARIMA models
    Returns predictions, metrics, and future forecasts
    """
    global current_data, prophet_model, arima_model, model_metrics
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Please upload data first.")
    
    try:
        # Prepare data
        if request.date_column not in current_data.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{request.date_column}' not found")
        
        if request.target_column not in current_data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        # Create time series dataframe
        df = current_data[[request.date_column, request.target_column]].copy()
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna().sort_values('date')
        
        # Aggregate by date if needed (sum values for same dates)
        df = df.groupby('date').agg({'value': 'sum'}).reset_index()
        
        # Split train/test
        split_idx = int(len(df) * request.train_test_split)
        train_data = df[:split_idx]
        test_data = df[split_idx:]
        
        if len(train_data) < 10:
            raise HTTPException(status_code=400, detail="Insufficient training data. Need at least 10 data points.")
        
        results = {
            'prophet': None,
            'arima': None
        }
        
        # ===== PROPHET MODEL =====
        if request.model_type in ['prophet', 'both']:
            # Prepare Prophet data
            prophet_train = train_data.copy()
            prophet_train.columns = ['ds', 'y']
            prophet_test = test_data.copy()
            prophet_test.columns = ['ds', 'y']
            
            # Train Prophet
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            prophet_model.fit(prophet_train)
            
            # Predict on test set
            prophet_forecast_test = prophet_model.predict(prophet_test[['ds']])
            
            # Future forecast
            future_periods = prophet_model.make_future_dataframe(periods=request.periods, freq='M')
            prophet_forecast_future = prophet_model.predict(future_periods)
            
            # Calculate metrics
            prophet_metrics = calculate_metrics(
                prophet_test['y'].values,
                prophet_forecast_test['yhat'].values
            )
            
            # Prepare predictions
            prophet_predictions = []
            for i in range(len(prophet_test)):
                prophet_predictions.append({
                    'date': prophet_test.iloc[i]['ds'].strftime('%Y-%m-%d'),
                    'actual': float(prophet_test.iloc[i]['y']),
                    'predicted': float(prophet_forecast_test.iloc[i]['yhat']),
                    'lower_bound': float(prophet_forecast_test.iloc[i]['yhat_lower']),
                    'upper_bound': float(prophet_forecast_test.iloc[i]['yhat_upper'])
                })
            
            # Prepare future forecast (only new periods)
            prophet_future = []
            future_only = prophet_forecast_future.tail(request.periods)
            for i in range(len(future_only)):
                prophet_future.append({
                    'date': future_only.iloc[i]['ds'].strftime('%Y-%m-%d'),
                    'predicted': float(future_only.iloc[i]['yhat']),
                    'lower_bound': float(future_only.iloc[i]['yhat_lower']),
                    'upper_bound': float(future_only.iloc[i]['yhat_upper'])
                })
            
            results['prophet'] = {
                'predictions': prophet_predictions,
                'metrics': prophet_metrics,
                'future_forecast': prophet_future
            }
        
        # ===== ARIMA MODEL =====
        if request.model_type in ['arima', 'both']:
            # Check stationarity
            stationarity = check_stationarity(train_data['value'])
            
            # Determine ARIMA parameters based on data size
            if len(train_data) < 20:
                arima_order = (1, 0, 1)
            else:
                arima_order = (1, 1, 1)
            
            # Train ARIMA
            arima_model = ARIMA(train_data['value'], order=arima_order)
            arima_fitted = arima_model.fit()
            
            # Predict on test set
            arima_forecast_test = arima_fitted.forecast(steps=len(test_data))
            
            # Future forecast
            arima_forecast_future = arima_fitted.forecast(steps=len(test_data) + request.periods)
            
            # Calculate metrics
            arima_metrics = calculate_metrics(
                test_data['value'].values,
                arima_forecast_test.values
            )
            arima_metrics['stationarity_pvalue'] = stationarity['p_value']
            arima_metrics['is_stationary'] = stationarity['is_stationary']
            
            # Prepare predictions
            arima_predictions = []
            for i in range(len(test_data)):
                arima_predictions.append({
                    'date': test_data.iloc[i]['date'].strftime('%Y-%m-%d'),
                    'actual': float(test_data.iloc[i]['value']),
                    'predicted': float(arima_forecast_test.iloc[i])
                })
            
            # Prepare future forecast
            arima_future = []
            future_dates = pd.date_range(
                start=df['date'].max(),
                periods=request.periods + 1,
                freq='M'
            )[1:]
            
            future_values = arima_forecast_future.values[-request.periods:]
            for i in range(len(future_dates)):
                arima_future.append({
                    'date': future_dates[i].strftime('%Y-%m-%d'),
                    'predicted': float(future_values[i])
                })
            
            results['arima'] = {
                'predictions': arima_predictions,
                'metrics': arima_metrics,
                'future_forecast': arima_future,
                'model_order': arima_order
            }
        
        # Store metrics globally
        model_metrics = results
        
        # Determine best model if both were used
        if request.model_type == 'both':
            prophet_rmse = results['prophet']['metrics']['RMSE']
            arima_rmse = results['arima']['metrics']['RMSE']
            best_model = 'prophet' if prophet_rmse < arima_rmse else 'arima'
            
            return ForecastResponse(
                model_used=best_model,
                forecast_periods=request.periods,
                predictions=results[best_model]['predictions'],
                metrics=results[best_model]['metrics'],
                future_forecast=results[best_model]['future_forecast']
            )
        else:
            model = request.model_type
            return ForecastResponse(
                model_used=model,
                forecast_periods=request.periods,
                predictions=results[model]['predictions'],
                metrics=results[model]['metrics'],
                future_forecast=results[model]['future_forecast']
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@app.get("/models/info")
def get_models_info():
    """Get information about trained models"""
    global model_metrics
    
    if not model_metrics:
        return {
            "message": "No models trained yet. Use /forecast endpoint to train models."
        }
    
    return {
        "models_trained": list(model_metrics.keys()),
        "model_metrics": model_metrics,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/models/compare")
def compare_models():
    """Compare performance of Prophet vs ARIMA if both are trained"""
    global model_metrics
    
    if 'prophet' not in model_metrics or 'arima' not in model_metrics:
        raise HTTPException(
            status_code=400,
            detail="Both models must be trained for comparison. Use model_type='both' in forecast request."
        )
    
    prophet_metrics = model_metrics['prophet']['metrics']
    arima_metrics = model_metrics['arima']['metrics']
    
    comparison = {
        'metrics_comparison': {
            'MAE': {
                'prophet': prophet_metrics['MAE'],
                'arima': arima_metrics['MAE'],
                'winner': 'prophet' if prophet_metrics['MAE'] < arima_metrics['MAE'] else 'arima'
            },
            'RMSE': {
                'prophet': prophet_metrics['RMSE'],
                'arima': arima_metrics['RMSE'],
                'winner': 'prophet' if prophet_metrics['RMSE'] < arima_metrics['RMSE'] else 'arima'
            },
            'MAPE': {
                'prophet': prophet_metrics['MAPE'],
                'arima': arima_metrics['MAPE'],
                'winner': 'prophet' if prophet_metrics['MAPE'] < arima_metrics['MAPE'] else 'arima'
            },
            'R2': {
                'prophet': prophet_metrics['R2'],
                'arima': arima_metrics['R2'],
                'winner': 'prophet' if prophet_metrics['R2'] > arima_metrics['R2'] else 'arima'
            }
        },
        'overall_winner': 'prophet' if prophet_metrics['RMSE'] < arima_metrics['RMSE'] else 'arima',
        'improvement_percentage': abs(
            (prophet_metrics['RMSE'] - arima_metrics['RMSE']) / 
            max(prophet_metrics['RMSE'], arima_metrics['RMSE'])
        ) * 100
    }
    
    return comparison

@app.get("/data/download")
def download_data():
    """Download current cleaned dataset as CSV"""
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available for download.")
    
    try:
        # Save to temporary file
        output_path = "/tmp/cleaned_finance_data.csv"
        current_data.to_csv(output_path, index=False)
        
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename="cleaned_finance_data.csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")

@app.delete("/data/clear")
def clear_data():
    """Clear current data and models"""
    global current_data, prophet_model, arima_model, model_metrics
    
    current_data = None
    prophet_model = None
    arima_model = None
    model_metrics = {}
    
    return {"message": "Data and models cleared successfully"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": current_data is not None,
        "models_trained": list(model_metrics.keys()) if model_metrics else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
