import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import optuna

def build_and_train_model(n_estimators, max_depth, min_samples_leaf, min_samples_split):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                  random_state=42)
    model.fit(X_train, y_train)
    return model

def optimize_hyperparameters(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
    
    model = build_and_train_model(n_estimators, max_depth, min_samples_leaf, min_samples_split)
    score = model.score(X_test, y_test)
    return score

# Read in the data from the CSV file
data = pd.read_csv('data/processed/ltpp_data.csv')

# Split the data into training and test sets
X = data.drop(columns=['IRI', 'STATION_ID'])
y = data['IRI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use optuna to tune the hyperparameters of the model
study = optuna.create_study(direction='maximize')
study.optimize(optimize_hyperparameters, n_trials=100)

best_params = study.best_params

# Train the final model using the best combination of hyperparameters
model = build_and_train_model(best_params['n_estimators'], best_params['max_depth'], 
                              best_params['min_samples_leaf'], best_params['min_samples_split'])

# Evaluate the model on the test set
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'R2: {r2:.2f}')

