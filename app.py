import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
st.title("Synthetic Data Generator")

# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'analyzed_params' not in st.session_state:
    st.session_state.analyzed_params = {}
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'date_column' not in st.session_state:
    st.session_state.date_column = None

# App tabs
tab1, tab2 = st.tabs(["CSV Data Generator", "Time Series Generator"])

with tab1:
    st.write("Upload your CSV file and generate synthetic data with the desired number of data points.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write("### Original Dataset Preview")
            st.write(df.head())
            
            # Display statistics
            st.write("### Dataset Statistics")
            st.write(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Data types information
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Numerical Columns")
                st.write(", ".join(numerical_cols) if numerical_cols else "None")
            with col2:
                st.write("#### Categorical Columns")
                st.write(", ".join(categorical_cols) if categorical_cols else "None")
            
            # Sample size slider
            st.write("### Generate Synthetic Data")
            sample_size = st.slider("Number of synthetic data points to generate:", 
                                  min_value=1, 
                                  max_value=max(10000, df.shape[0]*10), 
                                  value=df.shape[0],
                                  step=10)
            
            if st.button("Generate Synthetic Data", key="gen_csv_data"):
                with st.spinner("Generating synthetic data..."):
                    # Function to generate synthetic data
                    def generate_synthetic_data(data, n_samples):
                        synthetic_data = pd.DataFrame()
                        
                        for column in data.columns:
                            if data[column].dtype == 'object' or data[column].dtype == 'category':
                                # For categorical data, sample with replacement
                                values = data[column].fillna(data[column].mode()[0] if not data[column].mode().empty else "MISSING")
                                synthetic_data[column] = np.random.choice(values, size=n_samples, replace=True)
                            else:
                                # For numerical data, use a simple gaussian model with some noise
                                mean = data[column].mean()
                                std = data[column].std() if data[column].std() > 0 else 0.1
                                
                                # Handle NaN values
                                if np.isnan(mean) or np.isnan(std):
                                    synthetic_data[column] = np.random.choice(data[column].dropna(), size=n_samples, replace=True)
                                else:
                                    synthetic_values = np.random.normal(mean, std, n_samples)
                                    
                                    # For integer columns, round to integers
                                    if data[column].dtype in ['int32', 'int64']:
                                        synthetic_values = np.round(synthetic_values).astype(int)
                                    
                                    # Enforce min/max bounds
                                    min_val = data[column].min()
                                    max_val = data[column].max()
                                    synthetic_values = np.clip(synthetic_values, min_val, max_val)
                                    
                                    synthetic_data[column] = synthetic_values
                        
                        # Return both the synthetic data and its size
                        dataset_info = {
                            'data': synthetic_data,
                            'size': n_samples
                        }
                        
                        return dataset_info
                    
                    # Generate synthetic data
                    dataset_result = generate_synthetic_data(df, sample_size)
                    synthetic_df = dataset_result['data']
                    
                    # Display generated data
                    st.write("### Generated Synthetic Data Preview")
                    st.write(synthetic_df.head())
                    st.write(f"Shape: {synthetic_df.shape[0]} rows, {synthetic_df.shape[1]} columns")
                    st.write(f"Total dataset size: {dataset_result['size']} samples")
                    
                    # Validation Section
                    st.write("### Data Validation")
                    validation_tabs = st.tabs(["Statistical Comparison", "Distribution Comparison"])
                    
                    with validation_tabs[0]:
                        st.write("#### Statistical Comparison")
                        
                        # Compare statistics for numerical columns
                        if numerical_cols:
                            validation_stats = []
                            
                            for col in numerical_cols:
                                orig_mean = df[col].mean()
                                orig_std = df[col].std()
                                
                                syn_mean = synthetic_df[col].mean()
                                syn_std = synthetic_df[col].std()
                                
                                # Calculate relative difference in percentage
                                mean_diff_pct = abs(orig_mean - syn_mean) / (abs(orig_mean) if abs(orig_mean) > 0 else 1) * 100
                                std_diff_pct = abs(orig_std - syn_std) / (abs(orig_std) if abs(orig_std) > 0 else 1) * 100
                                
                                # Perform Kolmogorov-Smirnov test
                                ks_stat, ks_pval = stats.ks_2samp(df[col].dropna(), synthetic_df[col].dropna())
                                
                                validation_stats.append({
                                    "Column": col,
                                    "Original Mean": round(orig_mean, 2),
                                    "Synthetic Mean": round(syn_mean, 2),
                                    "Mean Diff %": round(mean_diff_pct, 2),
                                    "Original Std": round(orig_std, 2),
                                    "Synthetic Std": round(syn_std, 2),
                                    "Std Diff %": round(std_diff_pct, 2),
                                    "KS p-value": round(ks_pval, 4)
                                })
                            
                            # Display stats in a table
                            st.write(pd.DataFrame(validation_stats))
                            st.write("Note: Lower difference percentages indicate better similarity. KS p-value > 0.05 suggests similar distributions.")
                        
                        # Compare frequencies for categorical columns
                        if categorical_cols:
                            st.write("#### Categorical Column Validation")
                            selected_cat_col = st.selectbox("Select categorical column to compare:", categorical_cols)
                            
                            # Calculate frequencies
                            orig_counts = df[selected_cat_col].value_counts(normalize=True)
                            syn_counts = synthetic_df[selected_cat_col].value_counts(normalize=True)
                            
                            # Merge the frequencies
                            freq_comparison = pd.DataFrame({
                                'Original': orig_counts,
                                'Synthetic': syn_counts
                            }).fillna(0)
                            
                            # Calculate absolute difference
                            freq_comparison['Absolute Difference'] = abs(freq_comparison['Original'] - freq_comparison['Synthetic'])
                            
                            st.write(freq_comparison)
                    
                    with validation_tabs[1]:
                        st.write("#### Distribution Comparison")
                        
                        if numerical_cols:
                            selected_col = st.selectbox("Select column to compare:", numerical_cols, key="dist_compare")
                            
                            # Create a figure with two subplots
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Plot histograms
                            ax1.hist(df[selected_col].dropna(), bins=20, alpha=0.7)
                            ax1.set_title("Original Data")
                            ax1.set_xlabel(selected_col)
                            ax1.set_ylabel("Frequency")
                            
                            ax2.hist(synthetic_df[selected_col].dropna(), bins=20, alpha=0.7, color='orange')
                            ax2.set_title("Synthetic Data")
                            ax2.set_xlabel(selected_col)
                            ax2.set_ylabel("Frequency")
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Download link for synthetic data
                    csv = synthetic_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    filename = "synthetic_data.csv"
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Synthetic CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab2:
    st.write("Upload your CSV time series data and generate new synthetic data based on detected patterns")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file with time series data", type="csv", key="ts_uploader")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            data = pd.read_csv(uploaded_file)
            
            # Display dataframe preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Try to auto-detect date columns
            possible_date_columns = []
            for col in data.columns:
                try:
                    pd.to_datetime(data[col])
                    possible_date_columns.append(col)
                except:
                    pass
            
            if not possible_date_columns:
                st.warning("No column could be automatically detected as a date column. Please verify your data format.")
                possible_date_columns = data.columns.tolist()
            
            # Time Series Configuration
            st.subheader("Time Series Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                # Let user select which column contains the date
                date_column = st.selectbox(
                    "Select the column containing dates/times",
                    options=possible_date_columns,
                    index=0 if possible_date_columns else None
                )
            
            with col2:
                # Let user select which columns to analyze
                numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if not numeric_columns:
                    st.warning("No numeric columns detected. Please check your data format.")
                    numeric_columns = [col for col in data.columns if col != date_column]
                
                selected_columns = st.multiselect(
                    "Select time series columns to analyze",
                    options=numeric_columns,
                    default=numeric_columns[:1] if numeric_columns else None
                )
            
            if date_column and selected_columns:
                # Convert the selected column to datetime
                data[date_column] = pd.to_datetime(data[date_column])
                
                # Set the date column as index
                indexed_data = data.copy()
                indexed_data.set_index(date_column, inplace=True)
                indexed_data = indexed_data[selected_columns]
                
                # Store the uploaded data
                st.session_state.uploaded_data = indexed_data
                st.session_state.date_column = date_column
                
                # Plot the selected columns
                st.subheader("Original Time Series")
                fig, ax = plt.subplots(figsize=(10, 5))
                indexed_data.plot(ax=ax)
                ax.set_title("Original Time Series Data")
                ax.set_xlabel("Date/Time")
                ax.set_ylabel("Value")
                ax.legend(loc="best")
                st.pyplot(fig)
                
                # Extract time series properties for each selected column
                with st.spinner("Analyzing time series patterns..."):
                    analyzed_params = {}
                    
                    for column in selected_columns:
                        series = indexed_data[column].dropna()
                        
                        # Handle very short series
                        if len(series) < 4:
                            st.warning(f"Series '{column}' is too short for meaningful pattern analysis. Using default parameters.")
                            analyzed_params[column] = {
                                "freq": "D",
                                "include_trend": True,
                                "trend_strength": 1.0,
                                "trend_direction": "Increasing",
                                "include_seasonality": False,
                                "seasonality_strength": 0.0,
                                "seasonality_period": 30,
                                "noise_level": 1.0,
                                "min_value": float(series.min() if not series.empty else 0),
                                "max_value": float(series.max() if not series.empty else 100),
                                "mean_value": float(series.mean() if not series.empty else 50)
                            }
                            continue
                        
                        # Determine frequency
                        try:
                            freq = pd.infer_freq(series.index)
                            if freq is None:
                                # If can't infer, check median difference
                                diff = series.index.to_series().diff().median()
                                if diff.days == 1:
                                    freq = "D"  # Daily
                                elif diff.days == 7:
                                    freq = "W"  # Weekly
                                elif 28 <= diff.days <= 31:
                                    freq = "M"  # Monthly
                                else:
                                    freq = "D"  # Default to daily
                        except:
                            freq = "D"  # Default to daily
                        
                        # Detect seasonality period using autocorrelation
                        max_lag = min(len(series) // 2, 365)
                        seasonality_period = 30  # Default
                        if max_lag > 10:  # Only try if we have enough data
                            try:
                                autocorr = acf(series, nlags=max_lag)
                                # Find first significant peak after lag 1
                                threshold = 2 / np.sqrt(len(series))  # Significance threshold
                                for i in range(2, len(autocorr) - 1):
                                    if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                                        seasonality_period = i
                                        break
                            except:
                                pass
                        
                        # Calculate trend, seasonality, and noise using decomposition
                        try:
                            decomposition = seasonal_decompose(
                                series, 
                                model='additive', 
                                period=min(seasonality_period, len(series) // 2)
                            )
                            
                            trend = decomposition.trend.dropna()
                            trend_direction = "Increasing" if trend.iloc[-1] > trend.iloc[0] else "Decreasing"
                            trend_strength = abs(trend.max() - trend.min()) / (abs(series.max() - series.min()) or 1)
                            
                            seasonality = decomposition.seasonal.dropna()
                            seasonality_strength = seasonality.std() / (series.std() or 1)
                            
                            residual = decomposition.resid.dropna()
                            noise_level = residual.std() / (series.std() or 1)
                                
                        except:
                            # If decomposition fails, make simple estimates
                            trend_direction = "Increasing" if series.iloc[-1] > series.iloc[0] else "Decreasing"
                            trend_strength = abs(series.iloc[-1] - series.iloc[0]) / ((series.max() - series.min()) or 1)
                            seasonality_strength = 0.5  # Default
                            noise_level = 0.2  # Default
                        
                        analyzed_params[column] = {
                            "freq": freq,
                            "include_trend": trend_strength > 0.1,
                            "trend_strength": float(min(trend_strength * 10, 10.0)),
                            "trend_direction": trend_direction,
                            "include_seasonality": seasonality_strength > 0.1,
                            "seasonality_strength": float(min(seasonality_strength * 10, 10.0)),
                            "seasonality_period": int(seasonality_period),
                            "include_noise": True,
                            "noise_level": float(min(noise_level * 10, 5.0)),
                            "min_value": float(series.min()),
                            "max_value": float(series.max()),
                            "mean_value": float(series.mean())
                        }
                    
                    st.session_state.analyzed_params = analyzed_params
                
                # Display the detected time series patterns
                st.subheader("Detected Time Series Patterns")
                
                for column, params in analyzed_params.items():
                    with st.expander(f"Patterns for {column}", expanded=True):
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown("**Trend**")
                            trend_placeholder = st.empty()
                            trend_placeholder.progress(params['trend_strength'] / 10.0)
                            st.write(f"Direction: {params['trend_direction']}")
                        
                        with col2:
                            st.markdown("**Seasonality**")
                            if params['include_seasonality']:
                                season_placeholder = st.empty()
                                season_placeholder.progress(params['seasonality_strength'] / 10.0)
                                st.write(f"Period: ~{params['seasonality_period']} units")
                            else:
                                st.write("No significant seasonality detected")
                        
                        with col3:
                            st.markdown("**Noise**")
                            noise_placeholder = st.empty()
                            noise_placeholder.progress(params['noise_level'] / 5.0)
                
                # Configuration for generating new data
                st.subheader("Generate New Synthetic Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Date range for new data
                    st.markdown("**Time Range for New Data**")
                    
                    # Calculate sensible defaults for date range
                    if indexed_data.index.size > 0:
                        orig_start = indexed_data.index.min()
                        orig_end = indexed_data.index.max()
                        orig_duration = (orig_end - orig_start).days
                        
                        # Default to extending the existing time range forward by same duration
                        default_start = orig_end + timedelta(days=1)
                        default_end = default_start + timedelta(days=orig_duration)
                    else:
                        default_start = datetime.now()
                        default_end = default_start + timedelta(days=365)
                    
                    start_date = st.date_input("Start Date", value=default_start)
                    end_date = st.date_input("End Date", value=default_end)
                    
                    # Frequency matching
                    if analyzed_params:
                        detected_freq = next(iter(analyzed_params.values()))['freq']
                        freq_map = {
                            "D": "Daily",
                            "H": "Hourly",
                            "W": "Weekly", 
                            "M": "Monthly"
                        }
                        detected_freq_str = freq_map.get(detected_freq, "Daily")
                    else:
                        detected_freq_str = "Daily"
                    
                    freq = st.selectbox(
                        "Frequency",
                        options=["Daily", "Weekly", "Monthly"],
                        index=["Daily", "Weekly", "Monthly"].index(detected_freq_str) if detected_freq_str in ["Daily", "Weekly", "Monthly"] else 0
                    )
                
                with col2:
                    # Pattern adjustment
                    st.markdown("**Pattern Adjustment**")
                    
                    pattern_fidelity = st.slider(
                        "Pattern Fidelity",
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.7
                    )
                    
                    randomness = st.slider(
                        "Randomness",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3
                    )
                    
                    continuation = st.checkbox(
                        "Continuation Mode", 
                        value=True
                    )
                
                # Select series to generate
                series_to_generate = st.multiselect(
                    "Select series to generate", 
                    options=list(analyzed_params.keys()),
                    default=list(analyzed_params.keys())
                )
                
                # Generate button
                if st.button("Generate Time Series Data", type="primary", key="gen_ts_data"):
                    if not series_to_generate:
                        st.error("Please select at least one series to generate")
                    else:
                        with st.spinner("Generating synthetic time series data..."):
                            # Create dataframe with datetime index
                            all_series = pd.DataFrame()
                            
                            # Function to generate time series
                            def generate_time_series(start_date, end_date, freq_str, params, base_series=None, continuation=False):
                                # Convert frequency string to pandas frequency
                                freq_map = {
                                    "Daily": "D",
                                    "Weekly": "W",
                                    "Monthly": "MS"
                                }
                                pd_freq = freq_map[freq_str]
                                
                                # Generate date range
                                date_range = pd.date_range(start=start_date, end=end_date, freq=pd_freq)
                                n = len(date_range)
                                
                                if n == 0:
                                    st.error("No dates in selected range with specified frequency")
                                    return None
                                
                                # Setup components
                                t = np.arange(n)
                                
                                # Base level - starts with mean or continues from end
                                if continuation and base_series is not None and not base_series.empty:
                                    # Start from where the original series ended
                                    base_level = base_series.iloc[-1]
                                else:
                                    # Start from the mean of the original series
                                    base_level = params.get("mean_value", 0)
                                
                                # Trend component
                                if params["include_trend"]:
                                    # Scale trend strength by fidelity
                                    trend_str = params["trend_strength"] * pattern_fidelity
                                    
                                    if params["trend_direction"] == "Increasing":
                                        trend = trend_str * t / max(n, 1)
                                    else:
                                        trend = trend_str * (1 - t / max(n, 1))
                                else:
                                    trend = 0
                                
                                # Seasonality component
                                if params["include_seasonality"]:
                                    period = params["seasonality_period"]
                                    
                                    # Scale seasonality strength by fidelity
                                    season_str = params["seasonality_strength"] * pattern_fidelity
                                    
                                    # Adjust period for different frequencies
                                    adj_period = period / (1 if freq_str == "Daily" else 
                                                        7 if freq_str == "Weekly" else 
                                                        30 if freq_str == "Monthly" else 1)
                                    
                                    # If continuation mode, adjust phase to continue from original
                                    if continuation and base_series is not None and not base_series.empty:
                                        # Calculate where we are in the seasonal cycle
                                        orig_len = len(base_series)
                                        phase_shift = (orig_len % adj_period) / adj_period * 2 * np.pi
                                        seasonality = season_str * np.sin(2 * np.pi * t / adj_period + phase_shift)
                                    else:
                                        seasonality = season_str * np.sin(2 * np.pi * t / adj_period)
                                else:
                                    seasonality = 0
                                
                                # Noise component - scaled by randomness parameter
                                noise_level = params["noise_level"] * randomness
                                noise = noise_level * np.random.normal(0, 1, n)
                                
                                # Combine components
                                series = base_level + trend + seasonality + noise
                                
                                # Ensure values are within original min/max range
                                min_value = params.get("min_value", 0)
                                max_value = params.get("max_value", 100)
                                
                                # Allow exceeding original bounds slightly based on randomness
                                buffer = (max_value - min_value) * 0.1 * randomness
                                series = np.clip(series, min_value - buffer, max_value + buffer)
                                
                                return pd.Series(series, index=date_range)
                            
                            # Generate each selected series
                            for series_name in series_to_generate:
                                params = analyzed_params[series_name]
                                
                                # Get the original series for reference
                                if series_name in st.session_state.uploaded_data.columns:
                                    base_series = st.session_state.uploaded_data[series_name]
                                else:
                                    base_series = None
                                
                                # Generate the series
                                series = generate_time_series(
                                    start_date, 
                                    end_date, 
                                    freq, 
                                    params, 
                                    base_series,
                                    continuation
                                )
                                
                                if series is not None:
                                    all_series[series_name] = series
                            
                            if not all_series.empty:
                                # Cache the generated data
                                st.session_state.generated_data = all_series
                                
                                # Plot the data
                                st.subheader("Generated Synthetic Time Series")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                all_series.plot(ax=ax)
                                ax.set_title("Synthetic Time Series Based on Detected Patterns")
                                ax.set_xlabel("Date")
                                ax.legend(loc="best")
                                st.pyplot(fig)
                                
                                # Show relationship to original data
                                if continuation and not st.session_state.uploaded_data.empty:
                                    st.subheader("Original + Synthetic Data")
                                    
                                    # Create unified plot showing continuity
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    
                                    # Plot original data
                                    for col in st.session_state.uploaded_data.columns:
                                        if col in series_to_generate:
                                            st.session_state.uploaded_data[col].plot(
                                                ax=ax, 
                                                label=f"{col} (Original)",
                                                linestyle='-'
                                            )
                                    
                                    # Plot synthetic data
                                    for col in all_series.columns:
                                        all_series[col].plot(
                                            ax=ax, 
                                            label=f"{col} (Synthetic)",
                                            linestyle='--'
                                        )
                                    
                                    # Add a vertical line showing where original data ends and synthetic begins
                                    if not st.session_state.uploaded_data.empty:
                                        last_date = st.session_state.uploaded_data.index.max()
                                        ax.axvline(x=last_date, color='r', linestyle='--', alpha=0.7, 
                                                  label='End of Original Data')
                                    
                                    ax.set_title("Original + Synthetic Time Series")
                                    ax.set_xlabel("Date")
                                    ax.legend(loc="best")
                                    st.pyplot(fig)
                                
                                # Display dataframe preview
                                st.subheader("Generated Data Preview")
                                st.dataframe(all_series.head(10))
                                
                                # Download button
                                csv_buffer = io.BytesIO()
                                all_series.to_csv(csv_buffer)
                                csv_buffer.seek(0)
                                
                                st.download_button(
                                    label="Download Synthetic Data as CSV",
                                    data=csv_buffer,
                                    file_name=f"synthetic_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("Failed to generate synthetic data.")
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
    else:
        st.info("Upload a CSV file to begin analyzing and generating synthetic time series data")

# Sidebar with info
st.sidebar.title("About")
st.sidebar.info(
    """
    This app offers two tools for synthetic data generation:
    
    **CSV Data Generator:**
    - Create synthetic data preserving statistical properties
    - Works with both categorical and numerical data
    - Validates similarity between original and synthetic data
    
    **Time Series Generator:**
    - Extracts trends, seasonality, and patterns from time series
    - Generates continuation or new synthetic time series
    - Adjustable pattern fidelity and randomness
    
    These tools are useful for:
    - Expanding datasets for testing
    - Creating privacy-preserving synthetic data
    - Generating data for ML model training
    """
)