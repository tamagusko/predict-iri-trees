import pandas as pd

# Read and clean data
iri = pd.read_csv("data/raw/iri.csv")
iri = iri[["STATE_CODE", "SHRP_ID", "VISIT_DATE", "MRI"]]
iri["VISIT_DATE"] = pd.DatetimeIndex(iri["VISIT_DATE"]).year
iri.rename(columns={"VISIT_DATE": "Year"}, inplace=True)

traffic = pd.read_csv("data/raw/aadtt.csv")
traffic = traffic[["STATE_CODE", "SHRP_ID", "Year", "AADTT_ALL_TRUCKS_TREND"]]

sn = pd.read_csv("data/raw/sn.csv")
sn = sn[["STATE_CODE", "SHRP_ID", "SN_VALUE"]]

prec = pd.read_csv("data/raw/precipitation.csv")
prec = prec.drop(["STATE_CODE_EXP", "TOTAL_SNOWFALL_YR"], axis=1)

temp = pd.read_csv("data/raw/temperature.csv")
temp = temp.drop(["STATE_CODE_EXP", "FREEZE_THAW_YR", "FREEZE_INDEX_YR"], axis=1)

climate = pd.merge(
    prec, temp, on=["SHRP_ID", "STATE_CODE", "Year", "VWS_ID"]
).drop(["VWS_ID"], axis=1)

# Merge data
data = pd.merge(iri, climate, how="right", on=["SHRP_ID", "STATE_CODE", "Year"])
data = pd.merge(data, traffic, how="right", on=["SHRP_ID", "STATE_CODE", "Year"])
data = pd.merge(data, sn, how="right", on=["SHRP_ID", "STATE_CODE"])

# Create STATION_ID column and set it as the index
data["STATION_ID"] = data["SHRP_ID"].astype(str) + data["STATE_CODE"]
data.set_index("STATION_ID", inplace=True)

# Sort dataframe by STATION_ID and YEAR
data.sort_values(by=["STATION_ID", "Year"], inplace=True)

# Drop SHRP_ID and STATE_CODE columns and rename others
data.drop(columns=["SHRP_ID", "STATE_CODE"], inplace=True)
data = data.rename(
    {
        "MRI": "IRI",
        "TOTAL_ANN_PRECIP": "Precipitation",
        "MEAN_ANN_TEMP_AVG": "Temperature",
        "AADTT_ALL_TRUCKS_TREND": "AADTT",
        "SN_VALUE": "SN",
    },
    axis=1,
)

# Save the number of entries (raw)
number_raw_entries = data.shape[0]

# Drop duplicates and NaN values
data = data.drop_duplicates().dropna()

# Print report
print(f"Number of entries (raw data): {number_raw_entries}")
print(f"Number of outputs (after processing): {data.shape[0]}")

# Create data/processed/ folder if it does not exist
if not os.path.exists("data/processed/"):
    os.makedirs("data/processed/")

# Save data to CSV file
data.to_csv("data/processed/ltpp_data.csv", index=True, header=True)
