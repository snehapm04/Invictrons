import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Time Series Pattern Extractor", layout="wide")

st.title("Time Series Pattern Extractor & Synthetic Data Generator")
st.write("Upload your CSV time series data and generate new synthetic data based on detected patterns")

# Initialize session state variables if they don't exist
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'analyzed_params' not in st.session_state:
    st.session_state.analyzed_params = {}
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'date_column' not in st.session_state:
    st.session_state.date_column = None

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file with time series data", type="csv", help="Your file should contain at least one date/time column and one or more numeric columns")

if uploaded_file is not None:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        
        # Display dataframe preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Data Preview")
            st.dataframe(data.head())
        with col2:
            st.subheader("Dataset Info")
            st.write(f"Rows: {data.shape[0]}")
            st.write(f"Columns: {data.shape[1]}")
            st.write(f"Numeric columns: {len(data.select_dtypes(include=['float64', 'int64']).columns)}")
        
        # Identify potential date column
        st.subheader("Time Series Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            # Try to auto-detect date columns
            possible_date_columns = []
            for col in data.columns:
                # Try to convert the column to datetime
                try:
                    pd.to_datetime(data[col])
                    possible_date_columns.append(col)
                except:
                    pass
            
            if not possible_date_columns:
                st.warning("No column could be automatically detected as a date column. Please verify your data format.")
                possible_date_columns = data.columns.tolist()
            
            # Let user select which column contains the date
            date_column = st.selectbox(
                "Select the column containing dates/times",
                options=possible_date_columns,
                index=0 if possible_date_columns else None,
                help="This column will be used as the time index for analysis"
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
                default=numeric_columns[:1] if numeric_columns else None,
                help="These columns will be analyzed for patterns and used for synthetic data generation"
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
                            "include_cycles": False,
                            "cycle_strength": 0.0,
                            "cycle_period": 365,
                            "include_noise": True,
                            "noise_level": 1.0,
                            "include_outliers": False,
                            "outlier_percentage": 0.0,
                            "outlier_scale": 3.0,
                            "min_value": float(series.min() if not series.empty else 0),
                            "max_value": float(series.max() if not series.empty else 100),
                            "mean_value": float(series.mean() if not series.empty else 50)
                        }
                        continue
                    
                    # Compute frequency
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
                    if max_lag > 10:  # Only try if we have enough data
                        try:
                            autocorr = acf(series, nlags=max_lag)
                            # Find first significant peak after lag 1
                            threshold = 2 / np.sqrt(len(series))  # Significance threshold
                            for i in range(2, len(autocorr)):
                                if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                                    seasonality_period = i
                                    break
                            else:
                                seasonality_period = 30  # Default if no peak found
                        except:
                            seasonality_period = 30  # Default
                    else:
                        seasonality_period = 30  # Default for short series
                    
                    # Adjust seasonality period based on frequency
                    if freq == "W":
                        seasonality_period = max(4, seasonality_period // 7)  # Approx 4 weeks = monthly
                    elif freq == "M":
                        seasonality_period = max(3, seasonality_period // 30)  # Approx 3-4 months = quarterly
                    
                    # Ensure seasonality period is reasonable
                    seasonality_period = min(seasonality_period, len(series) // 4)
                    
                    # Calculate trend, seasonality, and noise using robust decomposition
                    try:
                        # Create a clean, resampled series for decomposition
                        clean_series = series.copy()
                        
                        # Try to decompose the series
                        decomposition = seasonal_decompose(
                            clean_series, 
                            model='additive', 
                            period=min(seasonality_period, len(clean_series) // 2)
                        )
                        
                        trend = decomposition.trend.dropna()
                        trend_direction = "Increasing" if trend.iloc[-1] > trend.iloc[0] else "Decreasing"
                        trend_strength = abs(trend.max() - trend.min()) / (abs(series.max() - series.min()) or 1)  # Avoid div by zero
                        
                        seasonality = decomposition.seasonal.dropna()
                        seasonality_strength = seasonality.std() / (series.std() or 1)  # Avoid div by zero
                        
                        residual = decomposition.resid.dropna()
                        noise_level = residual.std() / (series.std() or 1)  # Avoid div by zero
                            
                    except Exception as e:
                        # If decomposition fails, make simple estimates
                        trend_direction = "Increasing" if series.iloc[-1] > series.iloc[0] else "Decreasing"
                        trend_strength = abs(series.iloc[-1] - series.iloc[0]) / ((series.max() - series.min()) or 1)
                        seasonality_strength = 0.5  # Default
                        noise_level = 0.2  # Default
                    
                    # Detect outliers using Z-score
                    z_scores = np.abs(stats.zscore(series))
                    outliers = np.where(z_scores > 3)[0]
                    outlier_percentage = len(outliers) / len(series) * 100
                    outlier_scale = z_scores[outliers].mean() if len(outliers) > 0 else 3.0
                    
                    analyzed_params[column] = {
                        "freq": freq,
                        "include_trend": trend_strength > 0.1,
                        "trend_strength": float(min(trend_strength * 10, 10.0)),
                        "trend_direction": trend_direction,
                        "include_seasonality": seasonality_strength > 0.1,
                        "seasonality_strength": float(min(seasonality_strength * 10, 10.0)),
                        "seasonality_period": int(seasonality_period),
                        "include_cycles": False,  # Default
                        "cycle_strength": 0.0,
                        "cycle_period": 365,
                        "include_noise": True,
                        "noise_level": float(min(noise_level * 10, 5.0)),
                        "include_outliers": outlier_percentage > 1.0,  # Only if we have meaningful outliers
                        "outlier_percentage": float(min(outlier_percentage, 10.0)),
                        "outlier_scale": float(min(outlier_scale, 10.0)),
                        "min_value": float(series.min()),
                        "max_value": float(series.max()),
                        "mean_value": float(series.mean())
                    }
                
                st.session_state.analyzed_params = analyzed_params
                
                # Display the detected time series patterns in a visual way
                st.subheader("Detected Time Series Patterns")
                
                for i, (column, params) in enumerate(analyzed_params.items()):
                    with st.expander(f"Patterns for {column}", expanded=i==0):
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
                            st.markdown("**Noise & Outliers**")
                            noise_placeholder = st.empty()
                            noise_placeholder.progress(params['noise_level'] / 5.0)
                            if params['include_outliers']:
                                st.write(f"Outliers: {params['outlier_percentage']:.1f}% of data")
                            else:
                                st.write("No significant outliers detected")
                            
                        # Plot estimated components
                        try:
                            series = indexed_data[column].dropna()
                            period = min(params['seasonality_period'], len(series) // 2)
                            decomposition = seasonal_decompose(series, model='additive', period=period)
                            
                            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                            
                            # Original data
                            series.plot(ax=axes[0], color='blue')
                            axes[0].set_title('Original Data')
                            
                            # Trend
                            decomposition.trend.plot(ax=axes[1], color='red')
                            axes[1].set_title('Trend Component')
                            
                            # Seasonality
                            decomposition.seasonal.plot(ax=axes[2], color='green')
                            axes[2].set_title('Seasonal Component')
                            
                            # Residuals/Noise
                            decomposition.resid.plot(ax=axes[3], color='purple')
                            axes[3].set_title('Residual Component (Noise)')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        except:
                            st.write("Couldn't generate component plots - series may be too short or irregular.")
            
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
                
                # Check if dates are valid
                if start_date >= end_date:
                    st.error("End date must be after start date")
                
                # Frequency matching
                if 'freq' in next(iter(analyzed_params.values())):
                    detected_freq = next(iter(analyzed_params.values()))['freq']
                    freq_map = {
                        "D": "Daily",
                        "H": "Hourly",
                        "W": "Weekly", 
                        "M": "Monthly",
                        "Q": "Quarterly"
                    }
                    detected_freq_str = freq_map.get(detected_freq, "Daily")
                else:
                    detected_freq_str = "Daily"
                
                freq = st.selectbox(
                    "Frequency",
                    options=["Daily", "Hourly", "Weekly", "Monthly", "Quarterly"],
                    index=["Daily", "Hourly", "Weekly", "Monthly", "Quarterly"].index(detected_freq_str)
                )
            
            with col2:
                # Pattern adjustment
                st.markdown("**Pattern Adjustment**")
                
                pattern_fidelity = st.slider(
                    "Pattern Fidelity",
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.7,
                    help="Higher values make synthetic data more similar to original patterns"
                )
                
                randomness = st.slider(
                    "Randomness",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Higher values add more random variation to the synthetic data"
                )
                
                continuation = st.checkbox(
                    "Continuation Mode", 
                    value=True,
                    help="If checked, synthetic data will continue from the end of your actual data"
                )
            
            # Configure series to generate
            st.markdown("**Series Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select series to generate
                series_to_generate = st.multiselect(
                    "Select series to generate", 
                    options=list(analyzed_params.keys()),
                    default=list(analyzed_params.keys())
                )
            
            with col2:
                # How many copies of each series
                copies_per_series = st.number_input(
                    "Variations per series", 
                    min_value=1, 
                    max_value=5, 
                    value=1,
                    help="Generate multiple variations of each selected series"
                )
            
            # Generate button
            if st.button("Generate Synthetic Data", type="primary"):
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
                                "Hourly": "H",
                                "Weekly": "W",
                                "Monthly": "MS",
                                "Quarterly": "QS"
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
                                    trend = trend_str * t / max(n, 1)  # Avoid div by zero
                                else:
                                    trend = trend_str * (1 - t / max(n, 1))  # Avoid div by zero
                            else:
                                trend = 0
                            
                            # Seasonality component
                            if params["include_seasonality"]:
                                period = params["seasonality_period"]
                                
                                # Scale seasonality strength by fidelity
                                season_str = params["seasonality_strength"] * pattern_fidelity
                                
                                # Adjust period for different frequencies
                                adj_period = period / (1 if freq_str == "Daily" else 
                                                    24 if freq_str == "Hourly" else 
                                                    7 if freq_str == "Weekly" else 
                                                    30 if freq_str == "Monthly" else 90)
                                
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
                            
                            # Add outliers
                            if params["include_outliers"] and randomness > 0.1:
                                # Scale outlier frequency by randomness
                                outlier_pct = params["outlier_percentage"] * randomness
                                outlier_indices = np.random.choice(
                                    n, 
                                    size=int(n * outlier_pct / 100), 
                                    replace=False
                                )
                                outliers = params["outlier_scale"] * np.random.normal(0, 1, len(outlier_indices))
                                series[outlier_indices] += outliers
                            
                            # Ensure values are within original min/max range
                            min_value = params.get("min_value", 0)
                            max_value = params.get("max_value", 100)
                            
                            # Allow exceeding original bounds slightly based on randomness
                            buffer = (max_value - min_value) * 0.1 * randomness
                            series = np.clip(series, min_value - buffer, max_value + buffer)
                            
                            return pd.Series(series, index=date_range)
                        
                        # Generate each selected series and its variations
                        for series_name in series_to_generate:
                            params = analyzed_params[series_name]
                            
                            # Get the original series for reference
                            if series_name in st.session_state.uploaded_data.columns:
                                base_series = st.session_state.uploaded_data[series_name]
                            else:
                                base_series = None
                            
                            # Generate the main series
                            series = generate_time_series(
                                start_date, 
                                end_date, 
                                freq, 
                                params, 
                                base_series,
                                continuation
                            )
                            
                            if series is not None:
                                if copies_per_series == 1:
                                    # Just one copy, use original name
                                    all_series[series_name] = series
                                else:
                                    # Multiple variations
                                    all_series[f"{series_name}_1"] = series
                                    
                                    # Generate additional variations
                                    for i in range(2, copies_per_series + 1):
                                        # Create parameter variations
                                        var_params = params.copy()
                                        
                                        # Add some randomness to parameters
                                        if var_params["include_trend"]:
                                            var_params["trend_strength"] *= (0.8 + 0.4 * np.random.random())
                                        if var_params["include_seasonality"]:
                                            var_params["seasonality_strength"] *= (0.8 + 0.4 * np.random.random())
                                            var_params["seasonality_period"] *= (0.9 + 0.2 * np.random.random())
                                        
                                        # Generate with varied parameters
                                        var_series = generate_time_series(
                                            start_date, 
                                            end_date, 
                                            freq, 
                                            var_params, 
                                            base_series,
                                            continuation
                                        )
                                        
                                        if var_series is not None:
                                            all_series[f"{series_name}_{i}"] = var_series
                        
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
                                
                                # Plot first variation of each synthetic series
                                for col in all_series.columns:
                                    # Get base name (removing _1, _2, etc.)
                                    base_name = col.split('_')[0] if '_' in col and col.split('_')[-1].isdigit() else col
                                    if base_name in series_to_generate:
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
                                
                                ax.set_title("Original + Synthetic Time Series (Continuation Mode)")
                                ax.set_xlabel("Date")
                                ax.legend(loc="best")
                                st.pyplot(fig)
                            
                            # Display dataframe preview
                            st.subheader("Generated Data Preview")
                            st.dataframe(all_series.head(10))
                            
                            # Statistics about the generated data
                            with st.expander("Generated Data Statistics"):
                                st.dataframe(all_series.describe())
                            
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
                            st.error("Failed to generate synthetic data. Please check your parameters.")
            
            with st.expander("How this works"):
                st.markdown("""
                ### Pattern Detection & Synthesis Process
                
                1. **Pattern Analysis**: The app extracts key patterns from your time series data:
                   - **Trend**: Long-term upward or downward movement
                   - **Seasonality**: Regular, repeating patterns at fixed intervals
                   - **Noise**: Random variation in the data
                   - **Outliers**: Unusual data points significantly different from others
                
                2. **Synthetic Data Generation**:
                   - Uses detected patterns to create new synthetic data
                   - Maintains statistical properties of original data
                   - Can generate multiple variations of each series
                   
                3. **Continuation Mode**:
                   - When enabled, the synthetic data continues from where your original data ends
                   - Preserves trend direction and seasonal phase
                   - Useful for forecasting or extending existing datasets
                   
                4. **Adjustable Parameters**:
                   - **Pattern Fidelity**: Controls how closely synthetic data follows original patterns
                   - **Randomness**: Adds natural variation to make data realistic
                   
                5. **Applications**:
                   - Testing predictive models with expanded datasets
                   - Privacy-preserving data sharing
                   - Simulation of future scenarios
                   - Creating training data for machine learning
                """)
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
else:
    st.info("Upload a CSV file to begin analyzing and generating synthetic time series data")
    
    # Show example of what the app can do
    with st.expander("Example of what this app can do"):
        st.markdown("""
        ### Pattern Extraction & Synthetic Data Generation
        
        This app analyzes your time series CSV data and:
        
        1. **Automatically detects patterns** in your time series data:
           - Trend direction and strength
           - Seasonal patterns and their period
           - Noise levels and variability
           - Outlier characteristics
           
        2. **Generates new synthetic data** that follows those patterns:
           - Can extend your existing data into the future
           - Creates realistic variations with similar statistical properties
           - Preserves key characteristics of your original data
           
        3. **Provides visualization and comparison**:
           - Side-by-side comparison of original and synthetic data
           - Component breakdown (trend, seasonality, noise)
           - Statistical analysis
        
        4. **Exports synthetic data** as CSV for further use
        
        **Ideal for**: Expanding datasets for testing, simulation, training ML models, or creating privacy-preserving synthetic datasets.
        """)
        
        # Show example image
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Time-series.svg/1200px-Time-series.svg.png", 
                caption="Example of time series pattern analysis")