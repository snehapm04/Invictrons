import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io
import base64
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

st.title("CSV Synthetic Data Generator")
st.write("Upload your CSV file and generate synthetic data with the desired number of data points.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Original Dataset Preview")
        st.write(df.head())
        
        # Display statistics
        st.write("### Dataset Statistics")
        st.write(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data types information
        data_types = df.dtypes.astype(str)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
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
        
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                # Function to generate synthetic data
                def generate_synthetic_data(data, n_samples):
                    synthetic_data = pd.DataFrame()
                    
                    # Process each column separately
                    for column in data.columns:
                        if data[column].dtype == 'object' or data[column].dtype == 'category':
                            # For categorical data, sample with replacement
                            values = data[column].fillna(data[column].mode()[0] if not data[column].mode().empty else "MISSING")
                            synthetic_values = np.random.choice(values, size=n_samples, replace=True)
                            synthetic_data[column] = synthetic_values
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
                    
                    return synthetic_data
                
                # Generate synthetic data
                synthetic_df = generate_synthetic_data(df, sample_size)
                
                # Display generated data
                st.write("### Generated Synthetic Data Preview")
                st.write(synthetic_df.head())
                st.write(f"Shape: {synthetic_df.shape[0]} rows, {synthetic_df.shape[1]} columns")
                
                # Validation Section
                st.write("### Data Validation")
                
                # Create tabs for different validation methods
                validation_tabs = st.tabs(["Statistical Comparison", "Distribution Comparison", "Correlation Analysis"])
                
                with validation_tabs[0]:
                    st.write("#### Statistical Comparison")
                    
                    # Compare statistics for numerical columns
                    if numerical_cols:
                        validation_stats = []
                        
                        for col in numerical_cols:
                            # Calculate statistics for both original and synthetic data
                            orig_mean = df[col].mean()
                            orig_std = df[col].std()
                            orig_min = df[col].min()
                            orig_max = df[col].max()
                            
                            syn_mean = synthetic_df[col].mean()
                            syn_std = synthetic_df[col].std()
                            syn_min = synthetic_df[col].min()
                            syn_max = synthetic_df[col].max()
                            
                            # Calculate relative difference in percentage
                            mean_diff_pct = abs(orig_mean - syn_mean) / (abs(orig_mean) if abs(orig_mean) > 0 else 1) * 100
                            std_diff_pct = abs(orig_std - syn_std) / (abs(orig_std) if abs(orig_std) > 0 else 1) * 100
                            
                            # Perform Kolmogorov-Smirnov test to check distribution similarity
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
                        
                        # Calculate Chi-square test
                        # Create a contingency table
                        orig_vals = df[selected_cat_col].value_counts()
                        syn_vals = synthetic_df[selected_cat_col].value_counts()
                        
                        # Ensure both have the same categories
                        all_cats = sorted(set(orig_vals.index) | set(syn_vals.index))
                        orig_array = np.array([orig_vals.get(cat, 0) for cat in all_cats])
                        syn_array = np.array([syn_vals.get(cat, 0) for cat in all_cats])
                        
                        try:
                            chi2, p_val, _, _ = stats.chi2_contingency([orig_array, syn_array])
                            st.write(f"Chi-square p-value: {p_val:.4f} (p-value > 0.05 suggests similar distributions)")
                        except:
                            st.write("Couldn't calculate Chi-square test - may have insufficient data")
                
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
                        
                        # Q-Q plot to compare distributions - FIXED to handle small datasets
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Use replacement=True for small datasets and limit to 1000 samples maximum
                        orig_data = df[selected_col].dropna().values
                        syn_data = synthetic_df[selected_col].dropna().values
                        
                        sample_size = min(1000, len(orig_data))
                        
                        orig_sample = np.random.choice(orig_data, size=sample_size, replace=True)
                        stats.probplot(orig_sample, dist="norm", plot=ax)
                        ax.set_title(f"Q-Q Plot for {selected_col} (Original Data)")
                        st.pyplot(fig)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        syn_sample = np.random.choice(syn_data, size=sample_size, replace=True)
                        stats.probplot(syn_sample, dist="norm", plot=ax)
                        ax.set_title(f"Q-Q Plot for {selected_col} (Synthetic Data)")
                        st.pyplot(fig)
                
                with validation_tabs[2]:
                    st.write("#### Correlation Analysis")
                    
                    if len(numerical_cols) >= 2:
                        st.write("Correlation matrices comparison")
                        
                        # Calculate correlation matrices
                        orig_corr = df[numerical_cols].corr()
                        syn_corr = synthetic_df[numerical_cols].corr()
                        
                        # Create a figure with two subplots
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # Plot correlation heatmaps
                        im1 = ax1.imshow(orig_corr, cmap='coolwarm', vmin=-1, vmax=1)
                        ax1.set_title("Original Data Correlation")
                        ax1.set_xticks(np.arange(len(numerical_cols)))
                        ax1.set_yticks(np.arange(len(numerical_cols)))
                        ax1.set_xticklabels(numerical_cols, rotation=45, ha="right")
                        ax1.set_yticklabels(numerical_cols)
                        
                        im2 = ax2.imshow(syn_corr, cmap='coolwarm', vmin=-1, vmax=1)
                        ax2.set_title("Synthetic Data Correlation")
                        ax2.set_xticks(np.arange(len(numerical_cols)))
                        ax2.set_yticks(np.arange(len(numerical_cols)))
                        ax2.set_xticklabels(numerical_cols, rotation=45, ha="right")
                        ax2.set_yticklabels(numerical_cols)
                        
                        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Calculate difference between correlation matrices
                        corr_diff = abs(orig_corr - syn_corr)
                        st.write("#### Correlation Difference (Absolute)")
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(corr_diff, cmap='YlOrRd', vmin=0, vmax=2)
                        ax.set_title("Correlation Difference")
                        ax.set_xticks(np.arange(len(numerical_cols)))
                        ax.set_yticks(np.arange(len(numerical_cols)))
                        ax.set_xticklabels(numerical_cols, rotation=45, ha="right")
                        ax.set_yticklabels(numerical_cols)
                        
                        plt.colorbar(im, ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Average correlation difference
                        avg_corr_diff = corr_diff.mean().mean()
                        st.write(f"Average correlation difference: {avg_corr_diff:.4f} (lower is better)")
                    else:
                        st.write("Need at least 2 numerical columns for correlation analysis.")
                
                # Download link for synthetic data
                csv = synthetic_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                filename = "synthetic_data.csv"
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Synthetic CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
        # More detailed error information
        import traceback
        st.error(traceback.format_exc())
else:
    st.info("Please upload a CSV file to get started.")

# Add information section
st.sidebar.title("About")
st.sidebar.info(
    """
    This app generates synthetic data based on your uploaded CSV file. 
    
    The synthetic data generation:
    - Preserves statistical properties of numerical columns
    - Maintains distributions of categorical columns
    - Allows you to specify the number of synthetic samples
    - Provides validation with multiple metrics
    
    Key metrics:
    - Fidelity: How similar the synthetic data is to the original
    - Utility: How useful the synthetic data is for analysis
    
    For more complex needs or sensitive data, consider specialized tools.
    """
)
