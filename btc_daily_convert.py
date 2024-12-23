import pandas as pd

# Input and output file paths
input_file = "btcusd_1-min_data.csv"  # Replace with your file's name
output_file = "btcusd_daily_data.csv"

# Read the large CSV file
try:
    # Use pandas to read the CSV efficiently
    data = pd.read_csv(input_file)

    # Select every 1440th row
    daily_data = data.iloc[::1440, :]

    # Save the reduced data to a new CSV
    daily_data.to_csv(output_file, index=False)
    print(f"Processed file saved as {output_file}")
except FileNotFoundError:
    print(f"File {input_file} not found. Please check the path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
