"""
Finance ERP Streamlit Application
Interactive web app for financial data analysis and forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Prophet and ARIMA imports
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Finance ERP Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'prophet_model' not in st.session_state:
    st.session_state.prophet_model = None
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

# ==================== UTILITY FUNCTIONS ====================

def detect_date_columns(df):
    """Detect potential date columns in dataframe"""
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(), errors='coerce')
                if df[col].head().notna().sum() > 0:
                    date_cols.append(col)
            except:
                pass
    return date_cols

def handle_missing_values(df, method="ffill"):
    """Handle missing values using specified method"""
    if method == "Forward Fill":
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == "Backward Fill":
        return df.fillna(method='bfill').fillna(method='ffill')
    elif method == "Drop Rows":
        return df.dropna()
    elif method == "Mean":
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        return df.fillna(method='ffill')
    return df

def handle_outliers(df):
    """Handle outliers using IQR method"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_clean = df.copy()
    outliers_removed = {}
    
    for col in numeric_cols:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        before = len(df_clean)
        outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        outliers_removed[col] = outlier_mask.sum()
        
        df_clean[col] = df_clean[col].clip(lower, upper)
    
    return df_clean, outliers_removed

def check_stationarity(timeseries):
    """Check stationarity using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return {
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'is_stationary': result[1] <= 0.05,
        'critical_values': {k: float(v) for k, v in result[4].items()}
    }

def calculate_metrics(actual, predicted):
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

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("📊 Finance ERP")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Data", "🧹 Clean Data", "📈 Forecast", "📊 Model Comparison", "💾 Export Data"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Data status
    st.subheader("Data Status")
    if st.session_state.data is not None:
        st.success(f"✅ Data Loaded: {len(st.session_state.data)} rows")
    else:
        st.warning("⚠️ No data loaded")
    
    if st.session_state.cleaned_data is not None:
        st.success("✅ Data Cleaned")
    
    if st.session_state.forecast_results:
        st.success(f"✅ Models Trained: {len(st.session_state.forecast_results)}")
    
    st.markdown("---")
    
    # Clear data button
    if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
        st.session_state.data = None
        st.session_state.cleaned_data = None
        st.session_state.prophet_model = None
        st.session_state.arima_model = None
        st.session_state.forecast_results = {}
        st.rerun()

# ==================== HOME PAGE ====================

if page == "🏠 Home":
    st.title("📊 Finance ERP Analytics Platform")
    st.markdown("### Welcome to your comprehensive financial data analysis tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2);'>
            <h4>📤 Upload Data</h4>
            <p>Support for CSV and Excel files with automatic column detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2);'>
            <h4>🧹 Clean Data</h4>
            <p>Automated cleaning with outlier detection and missing value handling</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='padding: 20px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2);'>
            <h4>📈 Forecast</h4>
            <p>Prophet and ARIMA models for time series forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    with st.expander("1️⃣ Upload Your Data", expanded=True):
        st.markdown("""
        - Navigate to **Upload Data** from the sidebar
        - Upload your financial data (CSV or Excel format)
        - The app will automatically detect date and numeric columns
        - View basic statistics and data preview
        """)
    
    with st.expander("2️⃣ Clean Your Data"):
        st.markdown("""
        - Go to **Clean Data** section
        - Choose how to handle missing values
        - Enable outlier detection using IQR method
        - Normalize column names for consistency
        - Review cleaning summary and statistics
        """)
    
    with st.expander("3️⃣ Generate Forecasts"):
        st.markdown("""
        - Open **Forecast** section
        - Select target column (e.g., Sales, Profit)
        - Choose forecast model (Prophet, ARIMA, or Both)
        - Set forecast periods (e.g., 12 months)
        - View predictions and performance metrics
        """)
    
    with st.expander("4️⃣ Compare Models"):
        st.markdown("""
        - Visit **Model Comparison** section
        - Compare Prophet vs ARIMA performance
        - Analyze metrics (MAE, RMSE, MAPE, R²)
        - View side-by-side predictions
        """)
    
    st.markdown("---")
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✨ Key Features")
        st.markdown("""
        - **Data Pipeline**: Automated data loading and validation
        - **Smart Cleaning**: Handle missing values and outliers
        - **Multiple Models**: Prophet and ARIMA forecasting
        - **Interactive Visualizations**: Plotly charts for exploration
        - **Model Comparison**: Side-by-side performance analysis
        - **Export Options**: Download cleaned data and forecasts
        """)
    
    with col2:
        st.markdown("### 📊 Supported Models")
        st.markdown("""
        **Prophet (Facebook)**
        - Handles seasonality automatically
        - Provides confidence intervals
        - Robust to missing data
        - Best for seasonal patterns
        
        **ARIMA (Statistical)**
        - Classic time series model
        - Stationarity testing
        - Adaptive parameters
        - Best for stationary data
        """)

# ==================== UPLOAD DATA PAGE ====================

elif page == "📤 Upload Data":
    st.title("📤 Upload Financial Data")
    st.markdown("Upload your financial dataset (CSV or Excel format)")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
            
            # Basic info
            st.markdown("---")
            st.subheader("📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns)}")
            with col3:
                numeric_cols = df.select_dtypes(include=np.number).columns
                st.metric("Numeric Columns", f"{len(numeric_cols)}")
            with col4:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", f"{missing:,}")
            
            # Column info
            st.markdown("---")
            st.subheader("📊 Column Information")
            
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            
            st.dataframe(col_info, use_container_width=True, height=300)
            
            # Detect special columns
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔢 Numeric Columns")
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols:
                    for col in numeric_cols:
                        st.write(f"• {col}")
                else:
                    st.info("No numeric columns detected")
            
            with col2:
                st.subheader("📅 Date Columns")
                date_cols = detect_date_columns(df)
                if date_cols:
                    for col in date_cols:
                        st.write(f"• {col}")
                else:
                    st.info("No date columns detected")
            
            # Data preview
            st.markdown("---")
            st.subheader("👀 Data Preview")
            
            preview_rows = st.slider("Number of rows to display", 5, 50, 10)
            st.dataframe(df.head(preview_rows), use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("📈 Statistical Summary")
            
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns available for statistics")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a file to get started")
        
        # Show sample format
        st.markdown("---")
        st.subheader("📝 Expected Format")
        st.markdown("""
        Your data should include:
        - **Date column**: Date, Time, or similar naming
        - **Numeric columns**: Sales, Revenue, Profit, Units, etc.
        - **Regular time intervals**: Monthly, weekly, or daily
        
        Example structure:
        """)
        
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=12, freq='M'),
            'Sales': [100000, 105000, 110000, 108000, 115000, 120000, 
                     125000, 130000, 128000, 135000, 140000, 145000],
            'Profit': [20000, 21000, 22000, 21600, 23000, 24000,
                      25000, 26000, 25600, 27000, 28000, 29000]
        })
        
        st.dataframe(sample_data, use_container_width=True)

# ==================== CLEAN DATA PAGE ====================

elif page == "🧹 Clean Data":
    st.title("🧹 Data Cleaning")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        st.info("👉 Go to **Upload Data** section to get started")
    else:
        df = st.session_state.data.copy()
        
        st.markdown("Configure data cleaning options below:")
        
        # Cleaning options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Cleaning Options")
            
            missing_method = st.selectbox(
                "Handle Missing Values",
                ["Forward Fill", "Backward Fill", "Mean", "Drop Rows"],
                help="Choose how to handle missing values in the dataset"
            )
            
            handle_outliers_option = st.checkbox(
                "Handle Outliers (IQR Method)",
                value=True,
                help="Use Interquartile Range method to clip outliers"
            )
            
            normalize_cols = st.checkbox(
                "Normalize Column Names",
                value=True,
                help="Convert to lowercase and replace spaces with underscores"
            )
            
            convert_dates = st.checkbox(
                "Convert Date Columns",
                value=True,
                help="Automatically convert date columns to datetime format"
            )
        
        with col2:
            st.subheader("📊 Before Cleaning")
            st.metric("Total Rows", f"{len(df):,}")
            st.metric("Total Columns", f"{len(df.columns)}")
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
        
        st.markdown("---")
        
        # Clean data button
        if st.button("🧹 Clean Data", type="primary", use_container_width=True):
            with st.spinner("Cleaning data..."):
                cleaned_df = df.copy()
                operations = []
                
                # Handle missing values
                cleaned_df = handle_missing_values(cleaned_df, missing_method)
                operations.append(f"✓ Missing values handled using {missing_method}")
                
                # Handle outliers
                outliers_info = {}
                if handle_outliers_option:
                    cleaned_df, outliers_info = handle_outliers(cleaned_df)
                    operations.append(f"✓ Outliers handled using IQR method")
                
                # Normalize column names
                if normalize_cols:
                    cleaned_df.columns = cleaned_df.columns.str.strip().str.replace(" ", "_").str.lower()
                    operations.append("✓ Column names normalized")
                
                # Convert dates
                if convert_dates:
                    for col in cleaned_df.columns:
                        if 'date' in col.lower():
                            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    operations.append("✓ Date columns converted")
                
                # Save cleaned data
                st.session_state.cleaned_data = cleaned_df
                
                st.success("✅ Data cleaned successfully!")
                
                # Show operations
                st.markdown("### Operations Applied")
                for op in operations:
                    st.write(op)
                
                # Show outliers info
                if handle_outliers_option and outliers_info:
                    st.markdown("### Outliers Handled")
                    outlier_df = pd.DataFrame({
                        'Column': outliers_info.keys(),
                        'Outliers Clipped': outliers_info.values()
                    })
                    st.dataframe(outlier_df, use_container_width=True)
                
                # Comparison
                st.markdown("---")
                st.markdown("### 📊 Cleaning Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Rows After Cleaning",
                        f"{len(cleaned_df):,}",
                        delta=f"{len(cleaned_df) - len(df):,}"
                    )
                
                with col2:
                    st.metric(
                        "Missing Values After",
                        f"{cleaned_df.isnull().sum().sum():,}",
                        delta=f"-{df.isnull().sum().sum() - cleaned_df.isnull().sum().sum():,}"
                    )
                
                with col3:
                    st.metric(
                        "Columns After",
                        f"{len(cleaned_df.columns)}",
                        delta=f"{len(cleaned_df.columns) - len(df.columns)}"
                    )
        
        # Show current data
        if st.session_state.cleaned_data is not None:
            st.markdown("---")
            st.subheader("✨ Cleaned Data Preview")
            
            preview_rows = st.slider("Rows to display", 5, 50, 10, key="clean_preview")
            st.dataframe(st.session_state.cleaned_data.head(preview_rows), use_container_width=True)
            
            # Show statistics
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                st.markdown("### 📈 Statistics After Cleaning")
                st.dataframe(st.session_state.cleaned_data[numeric_cols].describe(), use_container_width=True)

# ==================== FORECAST PAGE ====================

elif page == "📈 Forecast":
    st.title("📈 Time Series Forecasting")
    
    # Use cleaned data if available, otherwise use raw data
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
        st.success("✅ Using cleaned data for forecasting")
    elif st.session_state.data is not None:
        df = st.session_state.data.copy()
        st.warning("⚠️ Using raw data. Consider cleaning first for better results.")
    else:
        st.warning("⚠️ Please upload data first!")
        st.info("👉 Go to **Upload Data** section to get started")
        st.stop()
    
    st.markdown("---")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Forecast Configuration")
        
        # Detect columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        date_cols = detect_date_columns(df)
        
        if not date_cols:
            st.error("❌ No date columns detected! Please ensure your data has a date column.")
            st.stop()
        
        if not numeric_cols:
            st.error("❌ No numeric columns detected! Please ensure your data has numeric values to forecast.")
            st.stop()
        
        date_col = st.selectbox("📅 Date Column", date_cols)
        target_col = st.selectbox("🎯 Target Column (to forecast)", numeric_cols)
        
        model_type = st.radio(
            "🤖 Select Model",
            ["Prophet", "ARIMA", "Both (Compare)"],
            help="Choose which forecasting model to use"
        )
        
        periods = st.slider(
            "📊 Forecast Periods",
            min_value=1,
            max_value=24,
            value=12,
            help="Number of future periods to forecast"
        )
        
        train_split = st.slider(
            "✂️ Train/Test Split",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Percentage of data to use for training"
        )
    
    with col2:
        st.subheader("📊 Data Overview")
        
        # Prepare time series data
        ts_df = df[[date_col, target_col]].copy()
        ts_df.columns = ['date', 'value']
        ts_df['date'] = pd.to_datetime(ts_df['date'])
        ts_df = ts_df.dropna().sort_values('date')
        ts_df = ts_df.groupby('date').agg({'value': 'sum'}).reset_index()
        
        st.metric("Total Data Points", len(ts_df))
        st.metric("Date Range", f"{ts_df['date'].min().date()} to {ts_df['date'].max().date()}")
        st.metric("Training Points", int(len(ts_df) * train_split))
        st.metric("Test Points", int(len(ts_df) * (1 - train_split)))
        
        # Quick visualization
        fig = px.line(ts_df, x='date', y='value', title=f"{target_col} Over Time")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Generate forecast button
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        
        if len(ts_df) < 10:
            st.error("❌ Insufficient data! Need at least 10 data points for forecasting.")
            st.stop()
        
        # Split data
        split_idx = int(len(ts_df) * train_split)
        train_data = ts_df[:split_idx]
        test_data = ts_df[split_idx:]
        
        results = {}
        
        # Prophet Model
        if model_type in ["Prophet", "Both (Compare)"]:
            with st.spinner("🔮 Training Prophet model..."):
                try:
                    # Prepare data
                    prophet_train = train_data.copy()
                    prophet_train.columns = ['ds', 'y']
                    prophet_test = test_data.copy()
                    prophet_test.columns = ['ds', 'y']
                    
                    # Train model
                    prophet_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    prophet_model.fit(prophet_train)
                    
                    # Predictions on test set
                    prophet_forecast_test = prophet_model.predict(prophet_test[['ds']])
                    
                    # Future forecast
                    future_periods = prophet_model.make_future_dataframe(periods=periods, freq='M')
                    prophet_forecast_future = prophet_model.predict(future_periods)
                    
                    # Metrics
                    prophet_metrics = calculate_metrics(
                        prophet_test['y'].values,
                        prophet_forecast_test['yhat'].values
                    )
                    
                    results['Prophet'] = {
                        'model': prophet_model,
                        'test_predictions': prophet_forecast_test,
                        'future_forecast': prophet_forecast_future,
                        'metrics': prophet_metrics,
                        'test_data': prophet_test
                    }
                    
                    st.session_state.prophet_model = prophet_model
                    st.success("✅ Prophet model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training Prophet: {str(e)}")
        
        # ARIMA Model
        if model_type in ["ARIMA", "Both (Compare)"]:
            with st.spinner("📊 Training ARIMA model..."):
                try:
                    # Check stationarity
                    stationarity = check_stationarity(train_data['value'])
                    
                    # Determine parameters
                    if len(train_data) < 20:
                        arima_order = (1, 0, 1)
                    else:
                        arima_order = (1, 1, 1)
                    
                    # Train model
                    arima_model = ARIMA(train_data['value'], order=arima_order)
                    arima_fitted = arima_model.fit()
                    
                    # Predictions
                    arima_forecast_test = arima_fitted.forecast(steps=len(test_data))
                    arima_forecast_future = arima_fitted.forecast(steps=len(test_data) + periods)
                    
                    # Metrics
                    arima_metrics = calculate_metrics(
                        test_data['value'].values,
                        arima_forecast_test.values
                    )
                    arima_metrics['stationarity_pvalue'] = stationarity['p_value']
                    arima_metrics['is_stationary'] = stationarity['is_stationary']
                    
                    # Future dates
                    future_dates = pd.date_range(
                        start=ts_df['date'].max(),
                        periods=periods + 1,
                        freq='M'
                    )[1:]
                    
                    results['ARIMA'] = {
                        'model': arima_fitted,
                        'test_predictions': arima_forecast_test,
                        'future_forecast': arima_forecast_future.values[-periods:],
                        'future_dates': future_dates,
                        'metrics': arima_metrics,
                        'test_data': test_data,
                        'order': arima_order
                    }
                    
                    st.session_state.arima_model = arima_fitted
                    st.success("✅ ARIMA model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training ARIMA: {str(e)}")
        
        # Save results
        st.session_state.forecast_results = results
        
        # Display results
        st.markdown("---")
        st.header("📊 Forecast Results")
        
        for model_name, result in results.items():
            st.subheader(f"📈 {model_name} Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = result['metrics']
            
            with col1:
                st.metric("MAE", f"{metrics['MAE']:,.2f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:,.2f}")
            with col3:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with col4:
                st.metric("R²", f"{metrics['R2']:.4f}")
            
            # Visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Test Set Predictions", "Future Forecast"),
                vertical_spacing=0.12
            )
            
            # Plot 1: Test predictions
            fig.add_trace(
                go.Scatter(x=train_data['date'], y=train_data['value'],
                          mode='lines+markers', name='Training Data',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            if model_name == 'Prophet':
                test_data_plot = result['test_data']
                predictions = result['test_predictions']
                
                fig.add_trace(
                    go.Scatter(x=test_data_plot['ds'], y=test_data_plot['y'],
                              mode='markers', name='Actual Test',
                              marker=dict(color='green', size=8)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=predictions['ds'], y=predictions['yhat'],
                              mode='lines+markers', name='Predicted',
                              line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                
                # Confidence interval
                fig.add_trace(
                    go.Scatter(x=predictions['ds'], y=predictions['yhat_upper'],
                              mode='lines', name='Upper Bound',
                              line=dict(width=0), showlegend=False),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=predictions['ds'], y=predictions['yhat_lower'],
                              mode='lines', name='Lower Bound',
                              line=dict(width=0), fillcolor='rgba(255,0,0,0.2)',
                              fill='tonexty', showlegend=False),
                    row=1, col=1
                )
                
                # Plot 2: Future forecast
                future = result['future_forecast'].tail(periods)
                fig.add_trace(
                    go.Scatter(x=ts_df['date'], y=ts_df['value'],
                              mode='lines', name='Historical',
                              line=dict(color='blue')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=future['ds'], y=future['yhat'],
                              mode='lines+markers', name='Future Forecast',
                              line=dict(color='purple', width=3)),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=future['ds'], y=future['yhat_upper'],
                              mode='lines', line=dict(width=0), showlegend=False),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=future['ds'], y=future['yhat_lower'],
                              mode='lines', line=dict(width=0),
                              fillcolor='rgba(128,0,128,0.2)', fill='tonexty',
                              showlegend=False),
                    row=2, col=1
                )
                
            else:  # ARIMA
                test_data_plot = result['test_data']
                predictions = result['test_predictions']
                
                fig.add_trace(
                    go.Scatter(x=test_data_plot['date'], y=test_data_plot['value'],
                              mode='markers', name='Actual Test',
                              marker=dict(color='green', size=8)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=test_data_plot['date'], y=predictions,
                              mode='lines+markers', name='Predicted',
                              line=dict(color='red', dash='dash')),
                    row=1, col=1
                )
                
                # Plot 2: Future forecast
                future_values = result['future_forecast']
                future_dates = result['future_dates']
                
                fig.add_trace(
                    go.Scatter(x=ts_df['date'], y=ts_df['value'],
                              mode='lines', name='Historical',
                              line=dict(color='blue')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=future_dates, y=future_values,
                              mode='lines+markers', name='Future Forecast',
                              line=dict(color='purple', width=3)),
                    row=2, col=1
                )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text=target_col, row=1, col=1)
            fig.update_yaxes(title_text=target_col, row=2, col=1)
            fig.update_layout(height=800, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Future forecast table
            st.markdown("#### 📋 Future Forecast Values")
            if model_name == 'Prophet':
                future_df = result['future_forecast'].tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                future_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                future_df['Date'] = pd.to_datetime(future_df['Date']).dt.date
                st.dataframe(future_df, use_container_width=True)
            else:
                future_df = pd.DataFrame({
                    'Date': [d.date() for d in result['future_dates']],
                    'Forecast': result['future_forecast']
                })
                st.dataframe(future_df, use_container_width=True)
            
            st.markdown("---")

# ==================== MODEL COMPARISON PAGE ====================

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    
    if not st.session_state.forecast_results:
        st.warning("⚠️ No models trained yet!")
        st.info("👉 Go to **Forecast** section and train models first")
    else:
        results = st.session_state.forecast_results
        
        if len(results) < 2:
            st.warning("⚠️ Please train both Prophet and ARIMA models for comparison")
            st.info("👉 Go to **Forecast** section and select 'Both (Compare)'")
        else:
            st.markdown("### Comparing Prophet vs ARIMA")
            
            prophet_metrics = results['Prophet']['metrics']
            arima_metrics = results['ARIMA']['metrics']
            
            # Metrics comparison
            st.subheader("📈 Performance Metrics")
            
            comparison_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'MAPE (%)', 'R²'],
                'Prophet': [
                    prophet_metrics['MAE'],
                    prophet_metrics['RMSE'],
                    prophet_metrics['MAPE'],
                    prophet_metrics['R2']
                ],
                'ARIMA': [
                    arima_metrics['MAE'],
                    arima_metrics['RMSE'],
                    arima_metrics['MAPE'],
                    arima_metrics['R2']
                ]
            })
            
            # Add winner column
            def determine_winner(row):
                if row['Metric'] == 'R²':
                    return 'Prophet' if row['Prophet'] > row['ARIMA'] else 'ARIMA'
                else:
                    return 'Prophet' if row['Prophet'] < row['ARIMA'] else 'ARIMA'
            
            comparison_df['Winner'] = comparison_df.apply(determine_winner, axis=1)
            
            # Style the dataframe
            def highlight_winner(row):
                if row['Winner'] == 'Prophet':
                    return ['', 'background-color: #90EE90', '', 'font-weight: bold']
                else:
                    return ['', '', 'background-color: #90EE90', 'font-weight: bold']
            
            styled_df = comparison_df.style.apply(highlight_winner, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Overall winner
            prophet_wins = (comparison_df['Winner'] == 'Prophet').sum()
            arima_wins = (comparison_df['Winner'] == 'ARIMA').sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prophet Wins", prophet_wins)
            with col2:
                st.metric("ARIMA Wins", arima_wins)
            with col3:
                overall_winner = 'Prophet' if prophet_wins > arima_wins else 'ARIMA'
                st.metric("🏆 Overall Winner", overall_winner)
            
            # Visualization comparison
            st.markdown("---")
            st.subheader("📊 Visual Comparison")
            
            # Bar chart of metrics
            fig = go.Figure()
            
            metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
            for metric in metrics_to_plot:
                row = comparison_df[comparison_df['Metric'] == metric].iloc[0] if metric != 'MAPE (%)' else comparison_df[comparison_df['Metric'] == 'MAPE (%)'].iloc[0]
                fig.add_trace(go.Bar(
                    name=metric,
                    x=['Prophet', 'ARIMA'],
                    y=[row['Prophet'], row['ARIMA']],
                    text=[f"{row['Prophet']:.2f}", f"{row['ARIMA']:.2f}"],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Metrics (Lower is Better)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # R² comparison
            fig2 = go.Figure()
            
            r2_row = comparison_df[comparison_df['Metric'] == 'R²'].iloc[0]
            fig2.add_trace(go.Bar(
                name='R²',
                x=['Prophet', 'ARIMA'],
                y=[r2_row['Prophet'], r2_row['ARIMA']],
                text=[f"{r2_row['Prophet']:.4f}", f"{r2_row['ARIMA']:.4f}"],
                textposition='auto',
                marker_color=['#636EFA', '#EF553B']
            ))
            
            fig2.update_layout(
                title="R² Score (Higher is Better)",
                height=400,
                yaxis_range=[0, 1]
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Side-by-side predictions
            st.markdown("---")
            st.subheader("🔄 Side-by-Side Predictions")
            
            prophet_test = results['Prophet']['test_data']
            prophet_pred = results['Prophet']['test_predictions']
            arima_test = results['ARIMA']['test_data']
            arima_pred = results['ARIMA']['test_predictions']
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=prophet_test['ds'],
                y=prophet_test['y'],
                mode='markers',
                name='Actual',
                marker=dict(size=10, color='green')
            ))
            
            fig3.add_trace(go.Scatter(
                x=prophet_pred['ds'],
                y=prophet_pred['yhat'],
                mode='lines+markers',
                name='Prophet Predictions',
                line=dict(color='blue', dash='dash')
            ))
            
            fig3.add_trace(go.Scatter(
                x=arima_test['date'],
                y=arima_pred.values,
                mode='lines+markers',
                name='ARIMA Predictions',
                line=dict(color='red', dash='dot')
            ))
            
            fig3.update_layout(
                title="Test Set Predictions Comparison",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            rmse_diff = abs(prophet_metrics['RMSE'] - arima_metrics['RMSE'])
            rmse_pct = (rmse_diff / max(prophet_metrics['RMSE'], arima_metrics['RMSE'])) * 100
            
            if overall_winner == 'Prophet':
                st.success(f"""
                ✅ **Prophet is recommended** for this dataset.
                
                - Prophet outperforms ARIMA in {prophet_wins} out of 4 metrics
                - RMSE is {rmse_pct:.1f}% better than ARIMA
                - Prophet handles seasonality well and provides confidence intervals
                - Best for: Datasets with strong seasonal patterns
                """)
            else:
                st.success(f"""
                ✅ **ARIMA is recommended** for this dataset.
                
                - ARIMA outperforms Prophet in {arima_wins} out of 4 metrics
                - RMSE is {rmse_pct:.1f}% better than Prophet
                - ARIMA provides more accurate short-term predictions
                - Best for: Stationary or near-stationary data
                """)

# ==================== EXPORT DATA PAGE ====================

elif page == "💾 Export Data":
    st.title("💾 Export Data")
    
    if st.session_state.cleaned_data is None and st.session_state.data is None:
        st.warning("⚠️ No data available for export!")
        st.info("👉 Please upload and clean data first")
    else:
        st.markdown("### Choose what to export")
        
        # Export cleaned data
        st.subheader("📊 Cleaned Dataset")
        
        if st.session_state.cleaned_data is not None:
            df_to_export = st.session_state.cleaned_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rows", f"{len(df_to_export):,}")
                st.metric("Columns", f"{len(df_to_export.columns)}")
            
            with col2:
                # CSV download
                csv = df_to_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"cleaned_finance_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Excel download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_to_export.to_excel(writer, index=False, sheet_name='Cleaned Data')
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"cleaned_finance_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.info("No cleaned data available. Clean your data first.")
        
        st.markdown("---")
        
        # Export forecast results
        st.subheader("📈 Forecast Results")
        
        if st.session_state.forecast_results:
            for model_name, result in st.session_state.forecast_results.items():
                st.markdown(f"#### {model_name} Forecasts")
                
                if model_name == 'Prophet':
                    future_df = result['future_forecast'].tail(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    future_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    
                    csv = future_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {model_name} Forecast (CSV)",
                        data=csv,
                        file_name=f"{model_name.lower()}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"{model_name}_csv"
                    )
                else:
                    future_df = pd.DataFrame({
                        'Date': result['future_dates'],
                        'Forecast': result['future_forecast']
                    })
                    
                    csv = future_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {model_name} Forecast (CSV)",
                        data=csv,
                        file_name=f"{model_name.lower()}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"{model_name}_csv"
                    )
                
                st.dataframe(future_df, use_container_width=True)
                st.markdown("---")
        else:
            st.info("No forecast results available. Generate forecasts first.")
        
        # Export metrics
        st.subheader("📊 Model Metrics")
        
        if st.session_state.forecast_results:
            metrics_data = []
            
            for model_name, result in st.session_state.forecast_results.items():
                metrics = result['metrics'].copy()
                metrics['Model'] = model_name
                metrics_data.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            cols = ['Model'] + [col for col in metrics_df.columns if col != 'Model']
            metrics_df = metrics_df[cols]
            
            st.dataframe(metrics_df, use_container_width=True)
            
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics (CSV)",
                data=csv,
                file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No metrics available. Generate forecasts first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Finance ERP Analytics Platform | Built with Streamlit</p>
    <p>📊 Data Pipeline • 🧹 Cleaning • 📈 Forecasting • 📊 Model Comparison</p>
</div>
""", unsafe_allow_html=True)