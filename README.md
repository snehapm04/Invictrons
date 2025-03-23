# Synthetic Data Generator

## Overview
A Streamlit web application that provides tools for generating synthetic data from CSV files and time series data. It allows users to create new datasets that preserve the statistical properties of the original data, which can be used for testing, privacy preservation, and machine learning model training.

## Features

### CSV Data Generator
- Upload CSV files and generate synthetic data with a specified number of samples
- Preserves statistical properties for both numerical and categorical data
- Provides validation tools to compare original and synthetic data:
  - Statistical comparison (mean, standard deviation, KS test)
  - Distribution comparison with visualizations
  - Categorical frequency analysis

### Time Series Generator
- Upload time series data and generate synthetic continuations or new series
- Automatically detects and extracts patterns:
  - Trend analysis (direction and strength)
  - Seasonality detection (period and strength)
  - Noise level estimation
- Customizable generation parameters:
  - Date range selection
  - Pattern fidelity adjustment
  - Randomness control
  - Continuation mode for seamless extensions
- Visualization of original and synthetic data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator

# Install required packages
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- Statsmodels

## Usage

```bash
# Run the application
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501

## How to Use

### CSV Data Generator
1. Select the "CSV Data Generator" tab
2. Upload a CSV file
3. View the original data preview and statistics
4. Select the number of synthetic data points to generate
5. Click "Generate Synthetic Data"
6. Explore the validation tabs to compare original and synthetic data
7. Download the generated data as a CSV file

### Time Series Generator
1. Select the "Time Series Generator" tab
2. Upload a CSV file containing time series data
3. Select the column containing dates/times
4. Choose the numerical columns to analyze
5. Review the detected patterns for each selected column
6. Set the time range and parameters for the new data:
   - Start and end dates
   - Frequency (Daily, Weekly, Monthly)
   - Pattern fidelity and randomness
   - Continuation mode (on/off)
7. Select which series to generate
8. Click "Generate Time Series Data"
9. View the visualizations and download the generated data

## Key Functions

### `generate_synthetic_data(data, n_samples)`
- Generates synthetic data based on the input data
- Parameters:
  - `data`: DataFrame containing the original data
  - `n_samples`: Number of synthetic samples to generate
- Returns: DataFrame containing the generated synthetic data and its size

### `generate_time_series(start_date, end_date, freq_str, params, base_series, continuation)`
- Generates synthetic time series data
- Parameters:
  - `start_date`: Start date for the generated series
  - `end_date`: End date for the generated series
  - `freq_str`: Frequency string ("Daily", "Weekly", "Monthly")
  - `params`: Dictionary of parameters extracted from original series
  - `base_series`: Original series (used for continuation)
  - `continuation`: Boolean flag for continuation mode
- Returns: Series object containing the generated time series

## License
[MIT License](LICENSE)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. 