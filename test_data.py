import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset with various data quality issues
n_samples = 1000

# Generate data
data = {
    # ID column (should be dropped)
    'customer_id': range(1, n_samples + 1),
    
    # Numerical features with different distributions
    'age': np.random.normal(35, 12, n_samples),
    'income': np.random.lognormal(10, 0.8, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples),
    'account_balance': np.random.normal(5000, 3000, n_samples),
    
    # Categorical features
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
    'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples, p=[0.7, 0.15, 0.15]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.45, 0.15]),
    
    # Binary features
    'owns_car': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'owns_house': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    
    # Feature with high correlation (multicollinearity issue)
    'monthly_income': None,  # Will be calculated based on income
    
    # Target variable (binary classification - loan approval)
    'loan_approved': None  # Will be calculated based on other features
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some realistic constraints
df['age'] = np.clip(df['age'], 18, 80)
df['credit_score'] = np.clip(df['credit_score'], 300, 850)
df['income'] = np.clip(df['income'], 20000, 200000)

# Create monthly income (highly correlated with income - multicollinearity issue)
df['monthly_income'] = df['income'] / 12 + np.random.normal(0, 200, n_samples)

# Create target variable based on logical rules
loan_probability = (
    (df['credit_score'] - 300) / 550 * 0.4 +  # Credit score effect
    (df['income'] - 20000) / 180000 * 0.3 +   # Income effect
    (df['age'] - 18) / 62 * 0.1 +             # Age effect
    df['owns_house'] * 0.1 +                  # House ownership effect
    (df['employment_status'] == 'Employed') * 0.1  # Employment effect
)

# Add some randomness and create binary target
df['loan_approved'] = (loan_probability + np.random.normal(0, 0.2, n_samples)) > 0.5
df['loan_approved'] = df['loan_approved'].astype(int)

# Introduce missing values in different patterns
# Random missing values
missing_indices_age = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
df.loc[missing_indices_age, 'age'] = np.nan

missing_indices_income = np.random.choice(df.index, size=int(0.08 * n_samples), replace=False)
df.loc[missing_indices_income, 'income'] = np.nan

missing_indices_education = np.random.choice(df.index, size=int(0.12 * n_samples), replace=False)
df.loc[missing_indices_education, 'education'] = np.nan

# Create a column with high missing values (should be dropped)
df['optional_field'] = np.random.normal(100, 20, n_samples)
high_missing_indices = np.random.choice(df.index, size=int(0.6 * n_samples), replace=False)
df.loc[high_missing_indices, 'optional_field'] = np.nan

# Add some outliers
outlier_indices = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
df.loc[outlier_indices, 'income'] *= 5  # Make some incomes extremely high
df.loc[outlier_indices, 'account_balance'] *= -3  # Make some balances extremely negative

# Create imbalanced target (more rejections than approvals)
# Flip some approvals to rejections to create imbalance
flip_indices = np.random.choice(df[df['loan_approved'] == 1].index, 
                               size=int(0.3 * len(df[df['loan_approved'] == 1])), 
                               replace=False)
df.loc[flip_indices, 'loan_approved'] = 0

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"Missing values per column:")
print(df.isnull().sum())
print(f"\nTarget distribution:")
print(df['loan_approved'].value_counts())
print(f"Class imbalance ratio: {df['loan_approved'].value_counts()[0] / df['loan_approved'].value_counts()[1]:.2f}:1")

# Save to CSV
df.to_csv('test_dataset.csv', index=False)
print("\nDataset saved as 'test_dataset.csv'")

# Display first few rows
print("\nFirst 10 rows:")
print(df.head(10))

# Show data types
print("\nData types:")
print(df.dtypes)