import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st


def show_removed_outliers(df_original: pd.DataFrame, df_pre_processed: pd.DataFrame):
    """
        This is a visualization function to compare the boxplot spread of the original dataset and 
        the processed dataset. 

        @parameters:

            df_original: pd.DataFrame
                The dataframe for the original dataset.

            df_preprocessed: pd.DataFrame
                The dataframe for the preprocessed dataset.

        @return:
            matplotlib.figure.Figure: The figure containing the plots
    """

    sns.set_style("whitegrid")

    # Get numerical columns
    numerical_cols = df_original.select_dtypes(include=['int64', 'float64']).columns

    # Define the figure and axes for the subplots
    fig, axes = plt.subplots(nrows=2, ncols=len(numerical_cols), figsize=(5*len(numerical_cols), 10))
    
    # Handle case where there's only one column
    if len(numerical_cols) == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    # Plot box-plots for original and pre-processed dataframe
    for i, column in enumerate(numerical_cols):
        # Original data (top row)
        sns.boxplot(data=df_original, y=column, ax=axes[0, i], palette="Set2")
        axes[0, i].set_title(f'Original - {column}')
        axes[0, i].set_xlabel('')
        
        # Processed data (bottom row)
        sns.boxplot(data=df_pre_processed, y=column, ax=axes[1, i], palette="Set3")
        axes[1, i].set_title(f'Pre-Processed - {column}')
        axes[1, i].set_xlabel('')

    # Adjust layout
    plt.tight_layout()
    return fig


def plot_filtered_correlation_matrix(df: pd.DataFrame):
    """
        This function plots the correlation matrix of the features based on Variance Inflation Factor ( VIF = 1/(1-R^2) ) 
        of each feature column. VIF is a strong indicator of multi-collinearity in our dataframe. 

        Note : The user is expected to remove some of the correlated features based on the VIF value.

        @parameter:

            df : pd.DataFrame
                The dataframe provided by user.
        
        @return:
            Tuple[matplotlib.figure.Figure, pd.DataFrame, List[str]]: 
                Figure, final correlation matrix, and messages about the process
    """
    numerical_columns = df.select_dtypes(include=['float64', 'int64'])
    non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
    messages = []
    messages.append(f"Before: {numerical_columns.shape}")
    
    vif = pd.DataFrame()
    vif['Feature'] = numerical_columns.columns
    vif["VIF"] = [variance_inflation_factor(numerical_columns.values, i) for i in range(numerical_columns.shape[1])]
    
    messages.append("Initial VIF values calculated")
    initial_vif = vif.sort_values(by="VIF", ascending=False).copy()
    
    infinite_vif_features = vif[vif["VIF"] == np.inf]["Feature"].tolist()
    if infinite_vif_features:
        messages.append(f"Dropping columns with infinite VIF values: {infinite_vif_features}")
        numerical_columns = numerical_columns.drop(columns=infinite_vif_features)
        
    max_vif = 10
    remove_flag = True
    removed_features = []
    
    while remove_flag and len(numerical_columns.columns) > 1:
        vif = pd.DataFrame()
        vif['Feature'] = numerical_columns.columns
        vif["VIF"] = [variance_inflation_factor(numerical_columns.values, i) for i in range(numerical_columns.shape[1])]
        
        max_vif_feature = vif.loc[vif['VIF'].idxmax()]
        
        if max_vif_feature['VIF'] > max_vif:
            numerical_columns = numerical_columns.drop(max_vif_feature['Feature'], axis=1)
            removed_features.append(f"{max_vif_feature['Feature']} (VIF={max_vif_feature['VIF']:.2f})")
            messages.append(f"Removed variable with high VIF {max_vif_feature['Feature']} (VIF={max_vif_feature['VIF']:.2f})")
        else:
            remove_flag = False

    messages.append(f"After: {numerical_columns.shape}")
    
    # Create the correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = numerical_columns.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='0.2g', cmap='coolwarm', 
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Correlation Matrix after VIF Filtering", fontsize=16, pad=20)
    plt.tight_layout()
    
    return fig, initial_vif, correlation_matrix, messages, removed_features