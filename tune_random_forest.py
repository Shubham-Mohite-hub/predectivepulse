import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Define categorical mappings
categorical_mappings = {
    'Stages': {0: 'HYPERTENSION (Stage-1)', 1: 'HYPERTENSION (Stage-2)',
               2: 'HYPERTENSIVE CRISIS', 3: 'NORMAL'}
}

# Print training data statistics
print("\nTraining Data Statistics:")
print(X_train.describe())
print("\nClass Distribution:")
print(pd.Series(y_train).value_counts().map(categorical_mappings['Stages']))

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)
best_rf = grid_search.best_estimator_
print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)

# Evaluate Random Forest
best_rf.fit(X_train_res, y_train_res)
y_pred = best_rf.predict(X_test)
y_test_decoded = pd.Series(y_test).map(categorical_mappings['Stages'])
y_pred_decoded = pd.Series(y_pred).map(categorical_mappings['Stages'])

# Compute and print metrics
print("\nResults for Random Forest:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=categorical_mappings['Stages'].values(),
                     columns=categorical_mappings['Stages'].values())
print("\nConfusion Matrix:")
print(cm_df)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_pred_proba = best_rf.predict_proba(X_test)
auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
print(f"\nAUC-ROC (One-vs-Rest): {auc:.4f}")

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test_decoded, 'Predicted': y_pred_decoded})
predictions_df.to_csv('random_forest_predictions.csv', index=False)
print("Predictions saved to random_forest_predictions.csv")

# Function to test custom input
def test_custom_input(model, categorical_mappings, feature_names):
    print("\nTesting custom input for Random Forest:")
    print("Categorical feature encodings:")
    print("Gender: 0 (Female), 1 (Male)")
    print("History, Patient, TakeMedication, BreathShortness, VisualChanges, NoseBleeding, ControlledDiet: 0 (No), 1 (Yes)")
    print("Severity: 0 (Mild), 1 (Moderate), 2 (Severe)")
    print("Whendiagnosed: 0 (1 - 5 Years), 1 (<1 Year), 2 (>5 Years)")
    print("Valid ranges: Age (18-100), Systolic (90-250), Diastolic (50-150)")

    custom_input = []
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
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
                elif feature == 'Age':
                    if not 18 <= value <= 100:
                        print("Age must be between 18 and 100.")
                        continue
                elif feature == 'Systolic':
                    if not 90 <= value <= 250:
                        print("Systolic must be between 90 and 250.")
                        continue
                elif feature == 'Diastolic':
                    if not 50 <= value <= 150:
                        print("Diastolic must be between 50 and 150.")
                        continue
                custom_input.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")

    custom_df = pd.DataFrame([custom_input], columns=feature_names)
    custom_pred = model.predict(custom_df)
    custom_pred_decoded = categorical_mappings['Stages'][custom_pred[0]]
    print(f"\nPredicted Hypertension Stage with Random Forest: {custom_pred_decoded}")

# Test custom input
test_custom_input(best_rf, categorical_mappings, X_train.columns)