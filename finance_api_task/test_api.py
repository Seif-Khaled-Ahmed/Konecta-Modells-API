"""
Test suite for Finance ERP API
Run with: pytest test_api.py -v
"""

import pytest
import requests
import pandas as pd
import io
from pathlib import Path

BASE_URL = "http://localhost:8000"

@pytest.fixture
def sample_csv_data():
    """Create sample financial data for testing"""
    data = {
        'Date': pd.date_range('2023-01-01', periods=24, freq='M'),
        'Sales': [100000 + i * 5000 for i in range(24)],
        'Profit': [20000 + i * 1000 for i in range(24)],
        'Units': [1000 + i * 50 for i in range(24)]
    }
    df = pd.DataFrame(data)
    
    # Save to CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer.getvalue().encode('utf-8')

@pytest.fixture
def upload_test_data(sample_csv_data):
    """Upload test data and return the response"""
    files = {'file': ('test_data.csv', sample_csv_data, 'text/csv')}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    return response

class TestHealthAndRoot:
    """Test basic endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

class TestDataUpload:
    """Test data upload functionality"""
    
    def test_upload_csv(self, sample_csv_data):
        """Test uploading CSV file"""
        files = {'file': ('test_data.csv', sample_csv_data, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "File uploaded successfully"
        assert "data_info" in data
        assert data["data_info"]["rows"] == 24
        assert data["data_info"]["columns"] == 4
    
    def test_upload_invalid_file(self):
        """Test uploading invalid file type"""
        files = {'file': ('test.txt', b'invalid data', 'text/plain')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        
        assert response.status_code == 400

class TestDataInfo:
    """Test data information endpoints"""
    
    def test_get_data_info_without_upload(self):
        """Test getting data info before uploading"""
        # Clear data first
        requests.delete(f"{BASE_URL}/data/clear")
        
        response = requests.get(f"{BASE_URL}/data/info")
        assert response.status_code == 400
    
    def test_get_data_info_with_data(self, upload_test_data):
        """Test getting data info after upload"""
        response = requests.get(f"{BASE_URL}/data/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "data_info" in data
        assert "statistics" in data
        assert len(data["data_info"]["numeric_columns"]) == 3

class TestDataCleaning:
    """Test data cleaning functionality"""
    
    def test_clean_data_default_config(self, upload_test_data):
        """Test cleaning with default configuration"""
        config = {
            "handle_missing": "ffill",
            "handle_outliers": True,
            "normalize_columns": True,
            "convert_dates": True
        }
        
        response = requests.post(f"{BASE_URL}/data/clean", json=config)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Data cleaned successfully"
        assert len(data["operations_applied"]) > 0
    
    def test_clean_data_different_methods(self, upload_test_data):
        """Test different cleaning methods"""
        methods = ["ffill", "bfill", "mean"]
        
        for method in methods:
            config = {
                "handle_missing": method,
                "handle_outliers": False,
                "normalize_columns": False,
                "convert_dates": False
            }
            
            response = requests.post(f"{BASE_URL}/data/clean", json=config)
            assert response.status_code == 200
    
    def test_clean_without_data(self):
        """Test cleaning before uploading data"""
        # Clear data first
        requests.delete(f"{BASE_URL}/data/clear")
        
        config = {
            "handle_missing": "ffill",
            "handle_outliers": True,
            "normalize_columns": True,
            "convert_dates": True
        }
        
        response = requests.post(f"{BASE_URL}/data/clean", json=config)
        assert response.status_code == 400

class TestForecasting:
    """Test forecasting functionality"""
    
    def test_forecast_prophet(self, upload_test_data):
        """Test Prophet forecasting"""
        # Clean data first
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": False,
            "normalize_columns": True,
            "convert_dates": True
        }
        requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        
        forecast_request = {
            "target_column": "sales",
            "date_column": "date",
            "periods": 6,
            "model_type": "prophet",
            "train_test_split": 0.8
        }
        
        response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "prophet"
        assert data["forecast_periods"] == 6
        assert "predictions" in data
        assert "metrics" in data
        assert "future_forecast" in data
        assert len(data["future_forecast"]) == 6
    
    def test_forecast_arima(self, upload_test_data):
        """Test ARIMA forecasting"""
        # Clean data first
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": False,
            "normalize_columns": True,
            "convert_dates": True
        }
        requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        
        forecast_request = {
            "target_column": "sales",
            "date_column": "date",
            "periods": 6,
            "model_type": "arima",
            "train_test_split": 0.8
        }
        
        response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "arima"
    
    def test_forecast_both_models(self, upload_test_data):
        """Test forecasting with both models"""
        # Clean data first
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": False,
            "normalize_columns": True,
            "convert_dates": True
        }
        requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        
        forecast_request = {
            "target_column": "sales",
            "date_column": "date",
            "periods": 6,
            "model_type": "both",
            "train_test_split": 0.8
        }
        
        response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] in ["prophet", "arima"]
        
        # Verify metrics are present
        assert "MAE" in data["metrics"]
        assert "RMSE" in data["metrics"]
        assert "MAPE" in data["metrics"]
        assert "R2" in data["metrics"]
    
    def test_forecast_invalid_column(self, upload_test_data):
        """Test forecasting with invalid column name"""
        forecast_request = {
            "target_column": "nonexistent_column",
            "date_column": "date",
            "periods": 6,
            "model_type": "prophet"
        }
        
        response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        assert response.status_code == 400

class TestModelComparison:
    """Test model comparison functionality"""
    
    def test_compare_models_without_training(self):
        """Test comparing models before training"""
        # Clear models
        requests.delete(f"{BASE_URL}/data/clear")
        
        response = requests.get(f"{BASE_URL}/models/compare")
        assert response.status_code == 400
    
    def test_compare_models_after_training(self, upload_test_data):
        """Test comparing models after training both"""
        # Clean and forecast with both models
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": False,
            "normalize_columns": True,
            "convert_dates": True
        }
        requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        
        forecast_request = {
            "target_column": "sales",
            "date_column": "date",
            "periods": 6,
            "model_type": "both"
        }
        requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        
        # Now compare
        response = requests.get(f"{BASE_URL}/models/compare")
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics_comparison" in data
        assert "overall_winner" in data
        assert "improvement_percentage" in data
        
        # Check that all metrics are compared
        assert "MAE" in data["metrics_comparison"]
        assert "RMSE" in data["metrics_comparison"]
        assert "MAPE" in data["metrics_comparison"]
        assert "R2" in data["metrics_comparison"]

class TestModelInfo:
    """Test model information endpoints"""
    
    def test_models_info_no_training(self):
        """Test getting model info before training"""
        requests.delete(f"{BASE_URL}/data/clear")
        
        response = requests.get(f"{BASE_URL}/models/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_models_info_after_training(self, upload_test_data):
        """Test getting model info after training"""
        # Clean and forecast
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": False,
            "normalize_columns": True,
            "convert_dates": True
        }
        requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        
        forecast_request = {
            "target_column": "sales",
            "date_column": "date",
            "periods": 6,
            "model_type": "both"
        }
        requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        
        # Get model info
        response = requests.get(f"{BASE_URL}/models/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "models_trained" in data
        assert len(data["models_trained"]) > 0

class TestDataDownload:
    """Test data download functionality"""
    
    def test_download_cleaned_data(self, upload_test_data):
        """Test downloading cleaned data"""
        # Clean data first
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": True,
            "normalize_columns": True,
            "convert_dates": True
        }
        requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        
        # Download
        response = requests.get(f"{BASE_URL}/data/download")
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/csv; charset=utf-8'
        
        # Verify it's valid CSV
        df = pd.read_csv(io.StringIO(response.text))
        assert len(df) > 0
    
    def test_download_without_data(self):
        """Test downloading before uploading data"""
        requests.delete(f"{BASE_URL}/data/clear")
        
        response = requests.get(f"{BASE_URL}/data/download")
        assert response.status_code == 400

class TestDataClear:
    """Test data clearing functionality"""
    
    def test_clear_data(self, upload_test_data):
        """Test clearing data and models"""
        response = requests.delete(f"{BASE_URL}/data/clear")
        
        assert response.status_code == 200
        assert response.json()["message"] == "Data and models cleared successfully"
        
        # Verify data is cleared
        info_response = requests.get(f"{BASE_URL}/data/info")
        assert info_response.status_code == 400

# Integration test
class TestCompleteWorkflow:
    """Test complete workflow from upload to forecast"""
    
    def test_full_workflow(self, sample_csv_data):
        """Test complete workflow"""
        # 1. Clear any existing data
        requests.delete(f"{BASE_URL}/data/clear")
        
        # 2. Upload data
        files = {'file': ('test_data.csv', sample_csv_data, 'text/csv')}
        upload_response = requests.post(f"{BASE_URL}/upload", files=files)
        assert upload_response.status_code == 200
        
        # 3. Get data info
        info_response = requests.get(f"{BASE_URL}/data/info")
        assert info_response.status_code == 200
        
        # 4. Clean data
        clean_config = {
            "handle_missing": "ffill",
            "handle_outliers": True,
            "normalize_columns": True,
            "convert_dates": True
        }
        clean_response = requests.post(f"{BASE_URL}/data/clean", json=clean_config)
        assert clean_response.status_code == 200
        
        # 5. Generate forecast
        forecast_request = {
            "target_column": "sales",
            "date_column": "date",
            "periods": 6,
            "model_type": "both"
        }
        forecast_response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        assert forecast_response.status_code == 200
        
        # 6. Compare models
        compare_response = requests.get(f"{BASE_URL}/models/compare")
        assert compare_response.status_code == 200
        
        # 7. Download data
        download_response = requests.get(f"{BASE_URL}/data/download")
        assert download_response.status_code == 200
        
        # 8. Clear data
        clear_response = requests.delete(f"{BASE_URL}/data/clear")
        assert clear_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
