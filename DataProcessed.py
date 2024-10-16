import pandas as pd
import ast

# Load the Excel file
file_path = 'hyderabad_cars.xlsx'  # Update this path with your file path
df = pd.read_excel(file_path)

# List of columns with JSON-like strings
json_columns = ['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs']


# Recursive function to flatten nested dictionaries and lists with column name in the key
def flatten_json(data, column_name='', parent_key='', sep='_'):
    items = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{column_name}{sep}{parent_key}{sep}{key}".strip(
                sep) if parent_key else f"{column_name}{sep}{key}"
            if isinstance(value, dict):
                # Recurse if value is a dictionary
                items.extend(flatten_json(value, column_name, new_key, sep=sep).items())
            elif isinstance(value, list):
                # If value is a list, expand each element
                for i, item in enumerate(value):
                    items.extend(flatten_json(item, column_name, f"{new_key}{sep}{i}", sep=sep).items())
            else:
                items.append((new_key, value))
    else:
        # If it's not a dictionary, return the data as is
        items.append((f"{column_name}{sep}{parent_key}".strip(sep), data))
    return dict(items)


# Parse and flatten each column with the correct column name in the key
flattened_data = []
for col in json_columns:
    # Parse JSON-like columns, only applying ast.literal_eval if the value is a string
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Apply the flatten_json function to the column and create a DataFrame
    flattened_col_df = df[col].apply(lambda x: pd.Series(flatten_json(x, column_name=col)))
    flattened_data.append(flattened_col_df)

# Combine all the expanded DataFrames
df_combined = pd.concat(flattened_data, axis=1)

# Save the structured DataFrame with expanded and flattened columns to an Excel file
output_file = 'structured_expanded_with_column_names_fixed.xlsx'  # Update with your desired output path
df_combined.to_excel(output_file, index=False)

print(f"File saved to: {output_file}")

files = ['Bangalore_Final.xlsx', 'Chennai_Final.xlsx', 'Delhi_Final.xlsx', 'Hyderabad_Final.xlsx', 'Jaipur_Final.xlsx',
         'Kolkata_Final.xlsx']

# Read and concatenate all files
dataframes = [pd.read_excel(file) for file in files]  # Reading all the Excel files into a list of DataFrames
concatenated_df = pd.concat(dataframes, ignore_index=True)  # Concatenating them into a single DataFrame
concatenated_df.to_excel(output_file, index=False)
print(f"File saved to: {output_file}")