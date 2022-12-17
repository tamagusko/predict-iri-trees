import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
import optuna


def build_and_train_model(depth, learning_rate, n_estimators):
    model = cb.CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def optimize_hyperparameters(trial):
    depth = trial.suggest_int("depth", 5, 15)
    learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.5)
    n_estimators = trial.suggest_int("n_estimators", 50, 200)

    model = build_and_train_model(depth, learning_rate, n_estimators)
    score = model.score(X_test, y_test)
    return score


# Read in the data from the CSV file
data = pd.read_csv("data/processed/ltpp_data.csv")

# Split the data into training and test sets
X = data.drop(columns=["IRI", "STATION_ID"])
y = data["IRI"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use optuna to tune the hyperparameters of the model
study = optuna.create_study(direction="maximize")
study.optimize(optimize_hyperparameters, n_trials=100)
# Hyperparameters report:
print("-----------------------")
print("Hyperparameters tunned!\n")

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train the final model using the best combination of hyperparameters
best_params = study.best_params
model = build_and_train_model(
    best_params["depth"], best_params["learning_rate"], best_params["n_estimators"]
)
