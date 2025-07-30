import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

# Load preprocessed data
X_train = pd.read_csv('./X_train.csv')
X_test = pd.read_csv('./X_test.csv')
y_train = pd.read_csv('./y_train.csv')
y_test = pd.read_csv('./y_test.csv')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
numerical_cols = ['Age', 'Systolic', 'Diastolic']
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Define hyperparameter grid for tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],  # Use liblinear to avoid saga convergence issues
    'max_iter': [2000]  # Increase max_iter for better convergence
}

# Initialize Logistic Regression model
logistic_regression = LogisticRegression()

# Perform GridSearchCV
grid_search = GridSearchCV(
    logistic_regression,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Train the model with hyperparameter tuning
grid_search.fit(X_train_scaled, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Predict on test data
y_pred = best_model.predict(X_test_scaled)

# Decode predictions and actuals for classification report
categorical_mappings = {
    'Stages': {0: 'HYPERTENSION (Stage-1)', 1: 'HYPERTENSION (Stage-2)',
               2: 'HYPERTENSIVE CRISIS', 3: 'NORMAL'}
}
y_test_decoded = pd.Series(y_test).map(categorical_mappings['Stages'])
y_pred_decoded = pd.Series(y_pred).map(categorical_mappings['Stages'])

# Evaluate model
score = accuracy_score(y_test, y_pred)
print("\nTest Accuracy Score:", score)
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Calculate ROC-AUC score (one-vs-rest for multiclass)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_proba = best_model.predict_proba(X_test_scaled)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
print("\nROC-AUC Score (One-vs-Rest):", roc_auc)

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test_decoded,
    'Predicted': y_pred_decoded
})
predictions_df.to_csv('tuned_logistic_predictions.csv', index=False)
print("\nPredictions saved to tuned_logistic_predictions.csv")

# Save the model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("\nModel saved to model.pkl")
print("Scaler saved to scaler.pkl")

# Function to test custom input
def test_custom_input(model, scaler, categorical_mappings, feature_names):
    print("\nEnter custom input for prediction (13 features):")
    print("Categorical feature encodings:")
    print("Gender: 0 (Female), 1 (Male)")
    print(
        "History, Patient, TakeMedication, BreathShortness, VisualChanges, NoseBleeding, ControlledDiet: 0 (No), 1 (Yes)")
    print("Severity: 0 (Mild), 1 (Moderate), 2 (Severe)")
    print("Whendiagnosed: 0 (1 - 5 Years), 1 (<1 Year), 2 (>5 Years)")

    # Collect input
    custom_input = []
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                # Validate categorical features
                if feature in ['Gender', 'History', 'Patient', 'TakeMedication',
                              'BreathShortness', 'VisualChanges', 'NoseBleeding',
                              'ControlledDiet']:
                    if value not in [0, 1]:
                        print(f"{feature} must be 0 or 1.")
                        continue
                elif feature == 'Severity':
                    if value not in [0, 1, 2]:
                        print("Severity must be 0, 1, or 2.")
                        continue
                elif feature == 'Whendiagnosed':
                    if value not in [0, 1, 2]:
                        print("Whendiagnosed must be 0, 1, or 2.")
                        continue
                custom_input.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")

    # Create DataFrame for input using training feature names
    custom_df = pd.DataFrame([custom_input], columns=feature_names)

    # Scale numerical features
    custom_df_scaled = custom_df.copy()
    custom_df_scaled[numerical_cols] = scaler.transform(custom_df[numerical_cols])

    # Predict
    custom_pred = model.predict(custom_df_scaled)
    custom_pred_decoded = categorical_mappings['Stages'][custom_pred[0]]

    # Predict probabilities
    custom_pred_proba = model.predict_proba(custom_df_scaled)[0]
    print(f"\nPredicted Hypertension Stage: {custom_pred_decoded}")
    print("Prediction Probabilities:")
    for stage, prob in zip(categorical_mappings['Stages'].values(), custom_pred_proba):
        print(f"{stage}: {prob:.4f}")

# Run custom input test using best model
test_custom_input(best_model, scaler, categorical_mappings, X_train.columns)
