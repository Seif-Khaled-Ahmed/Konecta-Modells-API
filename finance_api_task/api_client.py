"""
Sample Client for Finance ERP API
Demonstrates complete workflow: upload, clean, forecast
"""

import requests
import json
import sys
from typing import Dict, Any

class FinanceAPIClient:
    """Client for interacting with Finance ERP API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def upload_data(self, file_path: str) -> Dict[str, Any]:
        """Upload financial data file"""
        print(f"\n{'='*60}")
        print("UPLOADING DATA")
        print('='*60)
        
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    files={'file': f}
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"✓ File uploaded successfully: {file_path}")
                print(f"  Rows: {result['data_info']['rows']}")
                print(f"  Columns: {result['data_info']['columns']}")
                return result
                
        except FileNotFoundError:
            print(f"✗ Error: File not found - {file_path}")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"✗ Error uploading file: {e}")
            sys.exit(1)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about current dataset"""
        print(f"\n{'='*60}")
        print("DATA INFORMATION")
        print('='*60)
        
        try:
            response = requests.get(f"{self.base_url}/data/info")
            response.raise_for_status()
            result = response.json()
            
            info = result['data_info']
            print(f"  Dataset Shape: {info['rows']} rows × {info['columns']} columns")
            print(f"  Numeric Columns: {len(info['numeric_columns'])}")
            print(f"  Date Columns: {info['date_columns']}")
            
            # Show missing values if any
            missing = {k: v for k, v in info['missing_values'].items() if v > 0}
            if missing:
                print(f"  Missing Values: {len(missing)} columns with missing data")
            else:
                print("  Missing Values: None")
                
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error getting data info: {e}")
            return {}
    
    def clean_data(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Clean the dataset"""
        print(f"\n{'='*60}")
        print("CLEANING DATA")
        print('='*60)
        
        if config is None:
            config = {
                "handle_missing": "ffill",
                "handle_outliers": True,
                "normalize_columns": True,
                "convert_dates": True
            }
        
        try:
            response = requests.post(
                f"{self.base_url}/data/clean",
                json=config
            )
            response.raise_for_status()
            result = response.json()
            
            print("✓ Data cleaned successfully")
            print(f"  Operations applied:")
            for op in result['operations_applied']:
                print(f"    • {op}")
            print(f"  Remaining missing values: {result['remaining_missing']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error cleaning data: {e}")
            return {}
    
    def generate_forecast(
        self,
        target_column: str,
        date_column: str,
        periods: int = 12,
        model_type: str = "both"
    ) -> Dict[str, Any]:
        """Generate forecast"""
        print(f"\n{'='*60}")
        print("GENERATING FORECAST")
        print('='*60)
        
        forecast_request = {
            "target_column": target_column,
            "date_column": date_column,
            "periods": periods,
            "model_type": model_type,
            "train_test_split": 0.8
        }
        
        try:
            print(f"  Target: {target_column}")
            print(f"  Date Column: {date_column}")
            print(f"  Forecast Periods: {periods}")
            print(f"  Model Type: {model_type}")
            print("\n  Training models... (this may take a moment)")
            
            response = requests.post(
                f"{self.base_url}/forecast",
                json=forecast_request
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"\n✓ Forecast generated successfully")
            print(f"\n  Best Model: {result['model_used'].upper()}")
            print(f"\n  Performance Metrics:")
            print(f"    • MAE:  {result['metrics']['MAE']:,.2f}")
            print(f"    • RMSE: {result['metrics']['RMSE']:,.2f}")
            print(f"    • MAPE: {result['metrics']['MAPE']:.2f}%")
            print(f"    • R²:   {result['metrics']['R2']:.4f}")
            
            print(f"\n  Future Forecast (next {periods} periods):")
            for i, pred in enumerate(result['future_forecast'][:5], 1):
                print(f"    {i}. {pred['date']}: {pred['predicted']:,.2f}")
            
            if len(result['future_forecast']) > 5:
                print(f"    ... and {len(result['future_forecast']) - 5} more periods")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error generating forecast: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Details: {e.response.json().get('detail', 'Unknown error')}")
            return {}
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare Prophet vs ARIMA performance"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print('='*60)
        
        try:
            response = requests.get(f"{self.base_url}/models/compare")
            response.raise_for_status()
            result = response.json()
            
            print("\n  Metric Comparison:")
            for metric, values in result['metrics_comparison'].items():
                print(f"\n  {metric}:")
                print(f"    Prophet: {values['prophet']:,.2f}")
                print(f"    ARIMA:   {values['arima']:,.2f}")
                print(f"    Winner:  {values['winner'].upper()}")
            
            print(f"\n  Overall Winner: {result['overall_winner'].upper()}")
            print(f"  Improvement: {result['improvement_percentage']:.2f}%")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error comparing models: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Details: {e.response.json().get('detail', 'Unknown error')}")
            return {}
    
    def download_data(self, output_path: str = "cleaned_data.csv"):
        """Download cleaned dataset"""
        print(f"\n{'='*60}")
        print("DOWNLOADING DATA")
        print('='*60)
        
        try:
            response = requests.get(f"{self.base_url}/data/download")
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Data downloaded successfully: {output_path}")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error downloading data: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            result = response.json()
            
            print(f"\n{'='*60}")
            print("API HEALTH CHECK")
            print('='*60)
            print(f"  Status: {result['status'].upper()}")
            print(f"  Data Loaded: {result['data_loaded']}")
            print(f"  Models Trained: {result['models_trained']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"\n✗ API is not responding: {e}")
            print(f"  Make sure the API server is running at {self.base_url}")
            sys.exit(1)


def main():
    """Example usage of the Finance API Client"""
    
    # Initialize client
    client = FinanceAPIClient(base_url="http://localhost:8000")
    
    # Check API health
    client.health_check()
    
    # Example workflow - you'll need to provide your own data file
    FILE_PATH = "Financials.csv"  # Update this to your file path
    
    print("\n" + "="*60)
    print("FINANCE ERP API - COMPLETE WORKFLOW EXAMPLE")
    print("="*60)
    
    try:
        # 1. Upload data
        upload_result = client.upload_data(FILE_PATH)
        
        # 2. Get data information
        data_info = client.get_data_info()
        
        # Extract column names for forecast
        columns = data_info['data_info']['column_names']
        date_columns = data_info['data_info']['date_columns']
        numeric_columns = data_info['data_info']['numeric_columns']
        
        # 3. Clean data
        clean_result = client.clean_data()
        
        # 4. Generate forecast
        if date_columns and numeric_columns:
            # Use first date column and first numeric column
            date_col = date_columns[0]
            target_col = numeric_columns[0]
            
            forecast_result = client.generate_forecast(
                target_column=target_col,
                date_column=date_col,
                periods=12,
                model_type="both"
            )
            
            # 5. Compare models if both were trained
            if forecast_result:
                comparison = client.compare_models()
        else:
            print("\n⚠ Warning: No suitable date or numeric columns found for forecasting")
        
        # 6. Download cleaned data
        client.download_data("cleaned_finance_data.csv")
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # You can customize the workflow or call specific functions
    import argparse
    
    parser = argparse.ArgumentParser(description="Finance API Client")
    parser.add_argument('--file', type=str, help='Path to data file')
    parser.add_argument('--url', type=str, default='http://localhost:8000', 
                       help='API base URL')
    
    args = parser.parse_args()
    
    if args.file:
        client = FinanceAPIClient(base_url=args.url)
        client.health_check()
        client.upload_data(args.file)
        client.get_data_info()
        client.clean_data()
        print("\nData uploaded and cleaned. Use the API endpoints for forecasting.")
    else:
        # Run complete example workflow
        main()
