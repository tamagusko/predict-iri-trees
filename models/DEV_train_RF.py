import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import optuna

# Load the dataset
df = pd.read_csv("dataset.csv")

# Select the features and target
X = df[["Year", "Precipitation", "Temperature", "AADTT", "SN"]]
y = df["IRI"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(trial):
    # Set the hyperparameters for the model
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
    }

    # Train the model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    preds = model.predict(X_test)
    score = (preds - y_test).pow(2).mean()
    return score


# Use optuna to find the best hyperparameters
study = optuna.create_study()
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f"Best params: {best_params}")

# Train the model with the best hyperparameters
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

# Evaluate the model on the test set
score = model.score(X_test, y_test)
print(f"Test set R^2 score: {score:.2f}")

# Make predictions for IRI for 5 and 10 years for each STATION_ID
years = [5, 10]
predictions = {}
for station_id in df["STATION_ID"].unique():
    station_df = df[df["STATION_ID"] == station_id]
    current_year = station_df["Year"].max()
    for year in years:
        future_year = current_year + year
        X_pred = pd.DataFrame(
            {
                "Year": [future_year],
                "Precipitation": [station_df["Precipitation"].mean()],
                "Temperature": [station_df["Temperature"].mean()],
                "AADTT": [station_df["AADTT"].mean()],
                "SN": [station_df["SN"].mean()],
            }
        )
        y_pred = model.predict(X_pred)[0]
        predictions[(station_id, year)] = y_pred

print(predictions)
