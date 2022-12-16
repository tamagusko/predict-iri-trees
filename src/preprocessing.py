import os

import pandas as pd

# Read and clean data
iri = pd.read_csv('data/raw/iri.csv')
iri = iri[['STATE_CODE', 'SHRP_ID', 'VISIT_DATE', 'MRI']]
iri['YEAR'] = pd.DatetimeIndex(iri['VISIT_DATE']).year
iri = iri.drop(['VISIT_DATE'], axis=1)

traffic = pd.read_csv('data/raw/aadtt.csv')
traffic = traffic[['STATE_CODE', 'SHRP_ID', 'YEAR', 'AADTT_ALL_TRUCKS_TREND']]

sn = pd.read_csv('data/raw/sn.csv')
sn = sn[['STATE_CODE', 'SHRP_ID', 'SN_VALUE']]

prec = pd.read_csv('data/raw/precipitation.csv')
prec = prec.drop(['STATE_CODE_EXP', 'TOTAL_SNOWFALL_YR'], axis=1)

temp = pd.read_csv('data/raw/temperature.csv')
temp = temp.drop(
    [
        'STATE_CODE_EXP', 'FREEZE_THAW_YR',
        'FREEZE_INDEX_YR',
    ], axis=1,
)

climate = pd.merge(prec, temp, on=['SHRP_ID', 'STATE_CODE', 'YEAR', 'VWS_ID']).drop(
    ['VWS_ID'], axis=1,
)

# Merge data
data = pd.merge(
    iri, climate, how='right', on=[
        'SHRP_ID', 'STATE_CODE', 'YEAR',
    ],
)
data = pd.merge(
    data, traffic, how='right', on=[
        'SHRP_ID', 'STATE_CODE', 'YEAR',
    ],
)
data = pd.merge(data, sn, how='right', on=['SHRP_ID', 'STATE_CODE'])

# Create STATION_ID column and set it as the index
# STATION_ID = STATE_CODE + '_' + SHRP_ID
data['STATION_ID'] = data['STATE_CODE'].astype(
    str,
) + '_' + data['SHRP_ID'].astype(str)
data.set_index('STATION_ID', inplace=True)

# Drop SHRP_ID, STATE_CODE
data.drop(columns=['SHRP_ID', 'STATE_CODE'], inplace=True)

# Sort dataframe by STATION_ID and YEAR
data.sort_values(by=['STATION_ID', 'YEAR'], inplace=True)

# Rename Columns
data = data.rename(
    {
        'YEAR': 'Year',
        'MRI': 'IRI',
        'TOTAL_ANN_PRECIP': 'Precipitation',
        'MEAN_ANN_TEMP_AVG': 'Temperature',
        'AADTT_ALL_TRUCKS_TREND': 'AADTT',
        'SN_VALUE': 'SN',
    },
    axis=1,
)

# Save the number of entries (raw)
number_raw_entries = data.shape[0]

# Drop duplicates and NaN values
data = data.drop_duplicates().dropna()

# Round IRI, Temperature, and SN to 2 decimal places
data[['IRI', 'Temperature', 'SN']] = data[[
    'IRI', 'Temperature', 'SN',
]].round(2)

# Convert Precipitation to integer
data['Precipitation'] = data['Precipitation'].astype(int)

# Print report
print('Preprocessing report:\n')
print(f'Number of entries (raw data): {number_raw_entries}')
print(f'Number of outputs (after processing): {data.shape[0]}')

output = 'data/processed/ltpp_data.csv'
print(f'\nDataset saved on: {output}')

# Create data/processed/ folder if it does not exist
if not os.path.exists('data/processed/'):
    os.makedirs('data/processed/')

# Save data to CSV file
data.to_csv(output, index=True, header=True)
