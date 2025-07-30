import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import os

# Suppress convergence warnings for cleaner output (optional)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Ensure Flask directory exists
os.makedirs('Flask', exist_ok=True)

data = pd.read_csv('preprocessed_patient_data.csv')  # Update with your dataset path

# Define features and target
features = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity',
            'BreathShortness', 'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
            'Systolic', 'Diastolic', 'ControlledDiet']
target = 'Stages'  # Update with your target column name

# Verify all features exist
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

# Split features and target
X = data[features]
y = data[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
numerical_cols = ['Age', 'Systolic', 'Diastolic']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled = X_test.copy()
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Define Logistic Regression model and hyperparameter grid
model = LogisticRegression()
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],  # Use liblinear to avoid sag/saga convergence issues
    'max_iter': [1000]  # Increase max_iter to ensure convergence
}

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# Compute ROC-AUC (for multiclass, use one-vs-rest)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled), multi_class='ovr')
print(f"ROC-AUC score: {roc_auc:.4f}")

# Save model and scaler to Flask folder
with open('Flask/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('Flask/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved to Flask/model.pkl")
print("Scaler saved to Flask/scaler.pkl")