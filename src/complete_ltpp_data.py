import pandas as pd

# Read data from csv file
data = pd.read_csv('data/processed/ltpp_data_iri_verified.csv')

# Sort data by STATION_ID and Year
data.sort_values(by=['STATION_ID', 'Year'], inplace=True)

# Get number of initial rows
initial_rows = data.shape[0]

# Create a list to store the data frames for each STATION_ID
station_data_list = []

for station_id, group in data.groupby('STATION_ID'):
    # Get min and max year for this STATION_ID
    min_year = group['Year'].min()
    max_year = group['Year'].max()

    # Create a dataframe with all years for this STATION_ID
    station_years = pd.DataFrame({'Year': range(min_year, max_year + 1)})
    station_years['STATION_ID'] = station_id

    # Merge with original data
    station_data = pd.merge(
        station_years, group, on=[
            'STATION_ID', 'Year',
        ], how='left',
    )

    # Fill missing values with average for Precipitation, Temperature, and AADTT fields
    station_data['Precipitation'].fillna(
        station_data['Precipitation'].mean(), inplace=True,
    )
    station_data['Temperature'].fillna(
        station_data['Temperature'].mean(), inplace=True,
    )
    station_data['AADTT'].fillna(station_data['AADTT'].mean(), inplace=True)

    # Fill missing values with interpolated values for IRI and SN fields
    station_data['IRI'].interpolate(inplace=True)
    station_data['SN'].interpolate(inplace=True)

    # Add the data frame for this STATION_ID to the list
    station_data_list.append(station_data)

# Concatenate all the data frames in the list into a single data frame
result = pd.concat(station_data_list)

# Save resulting dataframe to csv file
output = 'data/processed/ltpp_data_final.csv'
result.to_csv('data/processed/ltpp_data_final.csv', index=False)

# Get number of final rows
final_rows = result.shape[0]

# Print number of initial and final rows
print('Processing and interpolating IRI report:\n')
print(f'Number of initial rows: {initial_rows}')
print(f'Number of final rows: {final_rows}')

print(f'\nDataset saved on: {output}')
