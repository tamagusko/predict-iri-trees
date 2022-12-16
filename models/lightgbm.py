import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna

def build_and_train_model(num_leaves, learning_rate, n_estimators):
    model = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
                              random_state=42)
    model.fit(X_train, y_train)
    return model

def optimize_hyperparameters(trial):
    num_leaves = trial.suggest_int('num_leaves', 30, 100)
    learning_rate = trial.suggest_uniform('learning_rate', 0.01, 0.5)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    
    model = build_and_train_model(num_leaves, learning_rate, n_estimators)
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
model = build_and_train_model(best_params['num_leaves'], best_params['learning_rate'], 
                              best_params['n_estimators'])

# Evaluate the model on the test set
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'R2: {r2:.2f}')

