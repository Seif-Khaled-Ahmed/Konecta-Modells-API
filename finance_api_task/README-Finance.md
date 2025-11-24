# Finance ERP API Service

A comprehensive REST API service that integrates financial data pipeline processing, data cleaning, and time series forecasting using Prophet and ARIMA models.

## Features

- 📊 **Data Upload & Management**: Upload CSV/Excel financial datasets
- 🧹 **Data Cleaning Pipeline**: Automated cleaning with configurable options
- 📈 **Time Series Forecasting**: Prophet and ARIMA models with performance comparison
- 📉 **Model Evaluation**: Comprehensive metrics (MAE, RMSE, MAPE, R²)
- 🔄 **Data Download**: Export cleaned datasets
- 🏥 **Health Monitoring**: Built-in health check endpoints

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python finance_api.py
```

Or using uvicorn directly:

```bash
uvicorn finance_api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Root - Get API Information
```http
GET /
```

**Response:**
```json
{
  "service": "Finance ERP API",
  "version": "1.0.0",
  "endpoints": {
    "upload": "/upload - Upload financial data",
    "info": "/data/info - Get dataset information",
    "clean": "/data/clean - Clean dataset",
    "forecast": "/forecast - Generate forecasts",
    "download": "/data/download - Download cleaned dataset",
    "models": "/models/info - Get model information"
  }
}
```

### 2. Upload Data
```http
POST /upload
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: File (CSV or Excel)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@Financials.csv"
```

**Example using Python:**
```python
import requests

with open('Financials.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
print(response.json())
```

**Response:**
```json
{
  "message": "File uploaded successfully",
  "data_info": {
    "filename": "Financials.csv",
    "rows": 700,
    "columns": 16,
    "column_names": ["Segment", "Country", "Product", "Date", "Sales", ...],
    "missing_values": {...},
    "sample_data": [...]
  }
}
```

### 3. Get Data Information
```http
GET /data/info
```

**Response:**
```json
{
  "data_info": {
    "rows": 700,
    "columns": 16,
    "column_names": [...],
    "missing_values": {...},
    "numeric_columns": ["Sales", "Units Sold", "Profit", ...],
    "date_columns": ["Date"]
  },
  "statistics": {...},
  "data_types": {...}
}
```

### 4. Clean Data
```http
POST /data/clean
```

**Request Body:**
```json
{
  "handle_missing": "ffill",
  "handle_outliers": true,
  "normalize_columns": true,
  "convert_dates": true
}
```

**Parameters:**
- `handle_missing`: Method to handle missing values
  - `"ffill"`: Forward fill (default)
  - `"bfill"`: Backward fill
  - `"drop"`: Drop rows with missing values
  - `"mean"`: Fill with column mean
- `handle_outliers`: Boolean - Use IQR method to handle outliers
- `normalize_columns`: Boolean - Normalize column names (lowercase, underscores)
- `convert_dates`: Boolean - Convert date columns to datetime

**Example:**
```python
import requests

config = {
    "handle_missing": "ffill",
    "handle_outliers": True,
    "normalize_columns": True,
    "convert_dates": True
}

response = requests.post('http://localhost:8000/data/clean', json=config)
print(response.json())
```

**Response:**
```json
{
  "message": "Data cleaned successfully",
  "operations_applied": [
    "Missing values handled using 'ffill' method",
    "Outliers handled using IQR method",
    "Column names normalized",
    "Date columns converted"
  ],
  "cleaned_rows": 700,
  "cleaned_columns": 16,
  "remaining_missing": 0
}
```

### 5. Generate Forecast
```http
POST /forecast
```

**Request Body:**
```json
{
  "target_column": "sales",
  "date_column": "date",
  "periods": 12,
  "model_type": "both",
  "train_test_split": 0.8
}
```

**Parameters:**
- `target_column`: Column name to forecast (e.g., "sales", "profit")
- `date_column`: Date column name
- `periods`: Number of future periods to forecast (default: 12)
- `model_type`: Model to use
  - `"prophet"`: Facebook Prophet model
  - `"arima"`: ARIMA model
  - `"both"`: Train both and return best performer (default)
- `train_test_split`: Train/test split ratio (default: 0.8)

**Example:**
```python
import requests

forecast_request = {
    "target_column": "sales",
    "date_column": "date",
    "periods": 12,
    "model_type": "both",
    "train_test_split": 0.8
}

response = requests.post('http://localhost:8000/forecast', json=forecast_request)
result = response.json()

print(f"Best Model: {result['model_used']}")
print(f"RMSE: {result['metrics']['RMSE']}")
print(f"R²: {result['metrics']['R2']}")
```

**Response:**
```json
{
  "model_used": "prophet",
  "forecast_periods": 12,
  "predictions": [
    {
      "date": "2023-10-01",
      "actual": 125000.50,
      "predicted": 123500.25,
      "lower_bound": 118000.00,
      "upper_bound": 129000.00
    },
    ...
  ],
  "metrics": {
    "MAE": 5234.67,
    "MSE": 38456789.45,
    "RMSE": 6201.35,
    "MAPE": 4.18,
    "R2": 0.87
  },
  "future_forecast": [
    {
      "date": "2024-01-01",
      "predicted": 130000.75,
      "lower_bound": 124000.00,
      "upper_bound": 136000.00
    },
    ...
  ]
}
```

### 6. Get Model Information
```http
GET /models/info
```

**Response:**
```json
{
  "models_trained": ["prophet", "arima"],
  "model_metrics": {
    "prophet": {
      "metrics": {
        "MAE": 5234.67,
        "RMSE": 6201.35,
        "MAPE": 4.18,
        "R2": 0.87
      }
    },
    "arima": {
      "metrics": {
        "MAE": 6789.12,
        "RMSE": 7845.23,
        "MAPE": 5.42,
        "R2": 0.82
      }
    }
  },
  "last_updated": "2024-01-15T10:30:45"
}
```

### 7. Compare Models
```http
GET /models/compare
```

**Response:**
```json
{
  "metrics_comparison": {
    "MAE": {
      "prophet": 5234.67,
      "arima": 6789.12,
      "winner": "prophet"
    },
    "RMSE": {
      "prophet": 6201.35,
      "arima": 7845.23,
      "winner": "prophet"
    },
    "MAPE": {
      "prophet": 4.18,
      "arima": 5.42,
      "winner": "prophet"
    },
    "R2": {
      "prophet": 0.87,
      "arima": 0.82,
      "winner": "prophet"
    }
  },
  "overall_winner": "prophet",
  "improvement_percentage": 20.95
}
```

### 8. Download Cleaned Data
```http
GET /data/download
```

**Example:**
```python
import requests

response = requests.get('http://localhost:8000/data/download')
with open('cleaned_data.csv', 'wb') as f:
    f.write(response.content)
```

### 9. Clear Data and Models
```http
DELETE /data/clear
```

**Response:**
```json
{
  "message": "Data and models cleared successfully"
}
```

### 10. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45",
  "data_loaded": true,
  "models_trained": ["prophet", "arima"]
}
```

## Complete Usage Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Upload data
print("1. Uploading data...")
with open('Financials.csv', 'rb') as f:
    response = requests.post(f"{BASE_URL}/upload", files={'file': f})
    print(json.dumps(response.json(), indent=2))

# 2. Get data info
print("\n2. Getting data info...")
response = requests.get(f"{BASE_URL}/data/info")
data_info = response.json()
print(f"Rows: {data_info['data_info']['rows']}")
print(f"Columns: {data_info['data_info']['columns']}")

# 3. Clean data
print("\n3. Cleaning data...")
clean_config = {
    "handle_missing": "ffill",
    "handle_outliers": True,
    "normalize_columns": True,
    "convert_dates": True
}
response = requests.post(f"{BASE_URL}/data/clean", json=clean_config)
print(json.dumps(response.json(), indent=2))

# 4. Generate forecast
print("\n4. Generating forecast...")
forecast_request = {
    "target_column": "sales",
    "date_column": "date",
    "periods": 12,
    "model_type": "both",
    "train_test_split": 0.8
}
response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
forecast_result = response.json()
print(f"Best Model: {forecast_result['model_used']}")
print(f"RMSE: {forecast_result['metrics']['RMSE']:.2f}")
print(f"MAPE: {forecast_result['metrics']['MAPE']:.2f}%")
print(f"R²: {forecast_result['metrics']['R2']:.3f}")

# 5. Compare models
print("\n5. Comparing models...")
response = requests.get(f"{BASE_URL}/models/compare")
comparison = response.json()
print(f"Overall Winner: {comparison['overall_winner']}")
print(f"Improvement: {comparison['improvement_percentage']:.2f}%")

# 6. Download cleaned data
print("\n6. Downloading cleaned data...")
response = requests.get(f"{BASE_URL}/data/download")
with open('cleaned_finance_data.csv', 'wb') as f:
    f.write(response.content)
print("Data downloaded successfully!")
```

## Model Information

### Prophet Model
- Facebook's time series forecasting model
- Handles seasonality automatically
- Provides confidence intervals
- Works well with missing data and outliers
- Best for data with strong seasonal patterns

### ARIMA Model
- Classic statistical model for time series
- ARIMA(p,d,q) where:
  - p: autoregressive order
  - d: differencing order
  - q: moving average order
- Automatically checks for stationarity
- Adapts parameters based on dataset size

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **MSE (Mean Squared Error)**: Average squared difference (penalizes large errors more)
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **R² (R-squared)**: Proportion of variance explained (0-1, higher is better)

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing data, invalid parameters)
- `500`: Server error (processing failures)

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Tips for Best Results

1. **Data Requirements**: 
   - Minimum 10 data points for training
   - 20+ recommended for ARIMA
   - Regular time intervals work best

2. **Data Cleaning**:
   - Always clean data before forecasting
   - Handle outliers if your data has extreme values
   - Normalize column names for consistency

3. **Model Selection**:
   - Use `"both"` to automatically select best model
   - Prophet works better with seasonal data
   - ARIMA works better with stationary data

4. **Forecasting**:
   - Don't forecast too far into the future (12 periods recommended)
   - Use 80/20 train/test split for evaluation
   - Check confidence intervals for uncertainty

## Production Deployment

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finance_api.py .

EXPOSE 8000

CMD ["uvicorn", "finance_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t finance-api .
docker run -p 8000:8000 finance-api
```

### Environment Variables

For production, consider adding:
```python
import os

PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
```

## Troubleshooting

**Issue**: "No data loaded" error
- **Solution**: Upload data first using `/upload` endpoint

**Issue**: "Insufficient training data" error
- **Solution**: Need at least 10 data points in your dataset

**Issue**: Model performance is poor
- **Solution**: 
  - Try both models and compare
  - Ensure data is cleaned properly
  - Check for sufficient historical data
  - Verify seasonality patterns

**Issue**: Forecast values seem unrealistic
- **Solution**:
  - Check for outliers in your data
  - Verify date column is properly formatted
  - Ensure target column has appropriate values

## License

This API service integrates the functionality from three Jupyter notebooks:
- ERP_Finance_data_pipeline.ipynb
- Finance_Cleaning_2.ipynb
- Finance_modelling_Final.ipynb

## Support

For issues or questions, please check:
1. API documentation at `/docs`
2. Health status at `/health`
3. Model information at `/models/info`
