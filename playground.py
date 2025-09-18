import pandas as pd

# Load the original dataset
df = pd.read_csv('tests_static\processed_Titanic.csv')

# Set a threshold for missing values. Let's say a row is "mostly empty"
# if it's missing 10 or more of its 12 values.
threshold = 10 

# Count the number of missing values (NaNs) in each row
missing_counts_per_row = df.isnull().sum(axis=1)

# Filter to find rows that exceed our threshold
mostly_empty_rows = df[missing_counts_per_row >= threshold]

if mostly_empty_rows.empty:
    print(f"No rows found with {threshold} or more missing values.")
else:
    print(f"Found rows with {threshold} or more missing values:")
    print(mostly_empty_rows)