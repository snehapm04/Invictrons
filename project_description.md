# Synthetic Data Generator

A powerful web application built with Streamlit for generating high-quality synthetic data from:
- CSV files (tabular data)
- Time series data

## Purpose

This tool helps data scientists, developers, and researchers create synthetic data that:
- Preserves statistical properties of the original data
- Maintains relationships between variables
- Produces realistic patterns without exposing sensitive information

## Use Cases

- Testing software with larger datasets
- ML model training when data is scarce
- Sharing non-sensitive versions of private data
- Creating demonstration datasets for presentations
- Extending time series for forecasting experiments

## Key Components

### CSV Data Generator
- Upload any CSV file to analyze its properties
- Generate synthetic data with customizable sample size
- Preserves distributions of numerical and categorical variables
- Built-in statistical validation tools for quality assessment

### Time Series Generator
- Advanced pattern detection (trend, seasonality, noise)
- Generates continuous or independent synthetic time series
- Adjustable parameters for pattern fidelity and randomness
- Multi-series support with interactive visualization

## Technical Features

- **Statistical Modeling**: Uses Gaussian models with appropriate constraints
- **Pattern Detection**: Implements autocorrelation and decomposition algorithms
- **Validation**: Includes Kolmogorov-Smirnov tests and distribution comparisons
- **Visualization**: Interactive charts for comparing original vs. synthetic data

## Implementation

Built with Python using:
- Streamlit (interactive UI)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib (visualization)
- SciPy & Statsmodels (statistical analysis)

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Access via browser at http://localhost:8501

## Benefits

- **Easy to use**: No coding required with intuitive UI
- **Flexible**: Works with various data types and formats
- **Transparent**: Explains detected patterns with visualizations
- **Customizable**: Fine-tune generation parameters for desired results
- **Validated**: Built-in tools to compare synthetic vs. original data 