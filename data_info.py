import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

def filtered_correlation_matrix(df: pd.DataFrame):
    """
        This function returns the Variance Inflation Factor ( VIF = 1/(1-R^2) ) of each feature column. VIF 
        is a strong indicator of multi-collinearity in our data frame. 

        Note : The user is expected to remove some of the correlated features based on the VIF value.

        @parameter:

            df : pd.DataFrame
                The dataframe provided by user.
        
        @return:
            tuple: (shape_info, vif_dataframe, messages)
    """

    numerical_columns = df.select_dtypes(include=['float64', 'int64'])
    messages = []
    messages.append(f"Before: {numerical_columns.shape}")
    
    vif = pd.DataFrame()
    vif['Feature'] = numerical_columns.columns
    vif["VIF"] = [variance_inflation_factor(numerical_columns.values, i) for i in range(numerical_columns.shape[1])]
    
    messages.append("Initial VIF values calculated")
    vif_sorted = vif.sort_values(by="VIF", ascending=False)
    
    return numerical_columns.shape, vif_sorted, messages


def display(file_path: str, df: pd.DataFrame) -> tuple:
    """
        The function returns the info of the original dataset in the given file path in comparison to
        the preprocessed dataframe.

        @parameters:
            file_path: str 
                The path to the original file.

            df: pd.DataFrame 
                The preprocessed dataframe.

        @return:
            tuple: (original_info, processed_info)
    """
    original_data = pd.read_csv(file_path)
    processed_data = df
    
    # Get info as strings
    import io
    import sys
    
    # Capture original data info
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    original_data.info()
    original_info = buffer.getvalue()
    
    # Capture processed data info
    sys.stdout = buffer = io.StringIO()
    processed_data.info()
    processed_info = buffer.getvalue()
    
    sys.stdout = old_stdout
    
    return original_info, processed_info