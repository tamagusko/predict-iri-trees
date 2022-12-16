import pandas as pd

# Read data from csv file
data = pd.read_csv("data/processed/ltpp_data.csv")

# Get number of initial rows
initial_rows = data.shape[0]

# Sort data by STATION_ID and Year
data.sort_values(by=["STATION_ID", "Year"], inplace=True)

# Delete rows where subsequent IRI value is less than previous one
data = data[data.groupby("STATION_ID")["IRI"].diff() >= 0]

# Save resulting dataframe to csv file
data.to_csv("data/processed/ltpp_data_iri_verified.csv", index=False)

# Get number of final rows
final_rows = data.shape[0]

# Print number of initial and final rows
print(f"Number of initial rows: {initial_rows}")
print(f"Number of final rows: {final_rows}")
