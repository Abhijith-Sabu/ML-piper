import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import tempfile
import os

# Import your custom modules (make sure they're in the same directory)
import data_info
import preprocess
import visualization

# Configure page
st.set_page_config(
    page_title="Data Preprocessing Pipeline",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🔧 Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Pipeline Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        try:
            # Load data for preview
            df = pd.read_csv(uploaded_file)
            
            st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            with st.expander("Column Information"):
                buffer = StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text(info_str)
            
            # Configuration
            st.sidebar.markdown("---")
            st.sidebar.header("Preprocessing Settings")
            
            # Target variable
            target_column = st.sidebar.selectbox(
                "Select Target Column", 
                options=df.columns.tolist(),
                help="The column you want to predict"
            )
            
            # Columns to drop
            dropped_columns = st.sidebar.multiselect(
                "Columns to Drop",
                options=[col for col in df.columns if col != target_column],
                help="Columns that should be completely removed (e.g., IDs)"
            )
            
            # Untouched columns
            available_cols = [col for col in df.columns if col not in dropped_columns and col != target_column]
            untouched_columns = st.sidebar.multiselect(
                "Untouched Columns",
                options=available_cols,
                help="Columns that should not be scaled or encoded"
            )
            
            # Dataset type
            type_dataset = st.sidebar.selectbox(
                "Dataset Type",
                options=["Cross Sectional", "Time Series"],
                help="Type of your dataset for appropriate imputation"
            )
            type_dataset = 0 if type_dataset == "Cross Sectional" else 1
            
            # Model type
            classification = st.sidebar.selectbox(
                "Model Type",
                options=["Classification", "Regression"],
                help="Type of model you plan to use"
            )
            classification = 1 if classification == "Classification" else 0
            
            # Sampling options
            if classification == 1:
                sampling = st.sidebar.checkbox("Apply Sampling for Imbalanced Data")
                if sampling:
                    strategy_sample = st.sidebar.selectbox(
                        "Sampling Strategy",
                        options=["smote", "oversampling", "undersampling"],
                        help="Strategy to handle imbalanced data"
                    )
                else:
                    strategy_sample = "smote"
            else:
                sampling = 0
                strategy_sample = "smote"
            
            # Process button
            if st.sidebar.button("🚀 Start Processing", type="primary"):
                process_data(temp_file_path, target_column, dropped_columns, untouched_columns,
                           type_dataset, int(sampling) if classification == 1 else 0, classification, strategy_sample, df, uploaded_file.name)
            
            # Analysis section
            st.markdown('<div class="section-header">📈 Data Analysis</div>', unsafe_allow_html=True)
            
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Basic Statistics", "Missing Values", "VIF Analysis"])
            
            with analysis_tab1:
                st.subheader("Descriptive Statistics")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Distribution plots
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    st.subheader("Distribution Plots")
                    selected_cols = st.multiselect("Select columns to plot", numerical_cols, default=numerical_cols[:3])
                    
                    if selected_cols:
                        fig, axes = plt.subplots(len(selected_cols), 2, figsize=(12, 4*len(selected_cols)))
                        if len(selected_cols) == 1:
                            axes = axes.reshape(1, -1)
                        
                        for i, col in enumerate(selected_cols):
                            # Histogram
                            axes[i, 0].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue')
                            axes[i, 0].set_title(f'Histogram - {col}')
                            axes[i, 0].set_xlabel(col)
                            axes[i, 0].set_ylabel('Frequency')
                            
                            # Box plot
                            axes[i, 1].boxplot(df[col].dropna())
                            axes[i, 1].set_title(f'Box Plot - {col}')
                            axes[i, 1].set_ylabel(col)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with analysis_tab2:
                st.subheader("Missing Values Analysis")
                missing_data = df.isnull().sum()
                missing_percent = (missing_data / len(df)) * 100
                
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': missing_percent.values
                }).sort_values('Missing Count', ascending=False)
                
                st.dataframe(missing_df, use_container_width=True)
                
                # Missing values heatmap
                if missing_data.sum() > 0:
                    st.subheader("Missing Values Heatmap")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
                    ax.set_title('Missing Values Heatmap')
                    st.pyplot(fig)
            
            with analysis_tab3:
                numerical_df = df.select_dtypes(include=['float64', 'int64'])
                if len(numerical_df.columns) > 1:
                    st.subheader("Variance Inflation Factor Analysis")
                    
                    try:
                        shape_info, vif_df, messages = data_info.filtered_correlation_matrix(df)
                        
                        st.write("**VIF Analysis Results:**")
                        for message in messages:
                            st.write(f"• {message}")
                        
                        st.subheader("VIF Values")
                        st.dataframe(vif_df, use_container_width=True)
                        
                        # Color code VIF values
                        def color_vif(val):
                            if val > 10:
                                return 'background-color: #ffcccb'
                            elif val > 5:
                                return 'background-color: #fff2cc'
                            else:
                                return 'background-color: #d4f6d4'
                        
                        styled_vif = vif_df.style.applymap(color_vif, subset=['VIF'])
                        st.dataframe(styled_vif, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Could not calculate VIF: {str(e)}")
                else:
                    st.info("Need at least 2 numerical columns for VIF analysis")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    else:
        st.info("👆 Please upload a CSV file to get started!")
        
        # Show example of what the app can do
        st.markdown('<div class="section-header">🎯 What This App Does</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Preprocessing Steps:**
            - 🔍 Initial data validation and cleaning
            - 🎯 Missing value imputation (mean/mode or interpolation)
            - ⚖️ Imbalanced data handling (SMOTE, oversampling, undersampling)
            - 📊 Outlier detection and removal
            - 🔤 Categorical encoding (Label/One-Hot)
            - 📏 Feature scaling (Standard/MinMax)
            """)
        
        with col2:
            st.markdown("""
            **Analysis Features:**
            - 📈 Descriptive statistics
            - 🔢 Missing value analysis
            - 🎯 VIF analysis for multicollinearity
            - 📊 Distribution plots
            - 📉 Before/after outlier comparison
            - 🔗 Correlation heatmaps
            """)


def process_data(file_path, target_column, dropped_columns, untouched_columns,
                type_dataset, sampling, classification, strategy_sample, original_df, original_filename=None):
    """Process the data using the preprocessing pipeline"""
    
    st.markdown('<div class="section-header">⚙️ Processing Results</div>', unsafe_allow_html=True)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Start processing
        status_text.text("Starting preprocessing pipeline...")
        progress_bar.progress(10)
        
        # Run the pipeline
        processed_df, messages = preprocess.preprocess_pipeline(
            file_path, target_column, dropped_columns, untouched_columns,
            type_dataset, sampling, classification, strategy_sample
        )
        
        progress_bar.progress(100)
        status_text.text("Processing completed!")
        
        # Show results
        st.markdown('<div class="success-box">✅ Processing completed successfully!</div>', unsafe_allow_html=True)
        
        # Processing log
        with st.expander("📋 Processing Log"):
            for message in messages:
                st.write(f"• {message}")
        
        # Results tabs
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "Processed Data", "Before vs After", "Visualizations", "Download"
        ])
        
        with result_tab1:
            st.subheader("Processed Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rows", original_df.shape[0])
            with col2:
                st.metric("Processed Rows", processed_df.shape[0])
            with col3:
                st.metric("Original Columns", original_df.shape[1])
            with col4:
                st.metric("Processed Columns", processed_df.shape[1])
            
            st.dataframe(processed_df.head(20), use_container_width=True)
            
            # Show data types
            st.subheader("Data Types After Processing")
            dtype_df = pd.DataFrame({
                'Column': processed_df.columns,
                'Data Type': processed_df.dtypes.astype(str),
                'Non-Null Count': processed_df.count(),
                'Null Count': processed_df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with result_tab2:
            st.subheader("Comparison: Original vs Processed")
            
            try:
                original_info, processed_info = data_info.display(file_path, processed_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original Data Info", original_info, height=300)
                with col2:
                    st.text_area("Processed Data Info", processed_info, height=300)
                
            except Exception as e:
                st.error(f"Could not generate comparison: {str(e)}")
        
        with result_tab3:
            st.subheader("Visualizations")
            
            # Outlier comparison
            try:
                numerical_cols_orig = original_df.select_dtypes(include=[np.number]).columns
                numerical_cols_proc = processed_df.select_dtypes(include=[np.number]).columns
                
                common_cols = list(set(numerical_cols_orig) & set(numerical_cols_proc))
                
                if len(common_cols) > 0:
                    st.write("**Outlier Removal Comparison**")
                    fig = visualization.show_removed_outliers(
                        original_df[common_cols], 
                        processed_df[common_cols]
                    )
                    st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not generate outlier comparison: {str(e)}")
            
            # Correlation matrix
            try:
                numerical_df = processed_df.select_dtypes(include=['float64', 'int64'])
                if len(numerical_df.columns) > 1:
                    st.write("**Correlation Matrix (VIF Filtered)**")
                    fig, initial_vif, final_corr, vif_messages, removed_features = visualization.plot_filtered_correlation_matrix(processed_df)
                    
                    st.pyplot(fig)
                    
                    if removed_features:
                        st.write("**Features removed due to high VIF:**")
                        for feature in removed_features:
                            st.write(f"• {feature}")
                    
                    with st.expander("VIF Processing Details"):
                        for message in vif_messages:
                            st.write(f"• {message}")
                
            except Exception as e:
                st.warning(f"Could not generate correlation matrix: {str(e)}")
        
        with result_tab4:
            st.subheader("Download Processed Data")
            
            # Convert to CSV
            csv = processed_df.to_csv(index=False)
            
            # Generate filename
            if original_filename is not None:
                filename = f"processed_{original_filename}"
            else:
                filename = "processed_data.csv"
                
            st.download_button(
                label="📥 Download Processed CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("Processing Summary")
            summary_stats = {
                "Original Shape": f"{original_df.shape[0]} rows × {original_df.shape[1]} columns",
                "Processed Shape": f"{processed_df.shape[0]} rows × {processed_df.shape[1]} columns",
                "Rows Removed": original_df.shape[0] - processed_df.shape[0],
                "Columns Removed": original_df.shape[1] - processed_df.shape[1],
                "Missing Values (Original)": original_df.isnull().sum().sum(),
                "Missing Values (Processed)": processed_df.isnull().sum().sum(),
            }
            
            summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
            st.table(summary_df)
    
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        st.write("Please check your configuration and try again.")


if __name__ == "__main__":
    main()