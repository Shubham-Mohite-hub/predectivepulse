import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
X_train = pd.read_csv('../X_train.csv')
X_test = pd.read_csv('../X_test.csv')
y_train = pd.read_csv('../y_train.csv')
y_test = pd.read_csv('../y_test.csv')

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

# Initialize Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Hyperparameter tuning with GridSearchCV
parameter = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_classifier = GridSearchCV(decision_tree, param_grid=parameter, scoring='accuracy', cv=5, n_jobs=-1)
dt_classifier.fit(X_train_scaled, y_train)

# Print best parameters and score
print("Best Parameters:", dt_classifier.best_params_)
print("Best Cross-Validation Score:", dt_classifier.best_score_)

# Predict on test data
y_pred = dt_classifier.predict(X_test_scaled)

# Decode predictions and actuals for classification report
categorical_mappings = {
    'Stages': {0: 'HYPERTENSION (Stage-1)', 1: 'HYPERTENSION (Stage-2)',
               2: 'HYPERTENSIVE CRISIS', 3: 'NORMAL'}
}
y_test_decoded = pd.Series(y_test).map(categorical_mappings['Stages'])
y_pred_decoded = pd.Series(y_pred).map(categorical_mappings['Stages'])

# Evaluate model
score = accuracy_score(y_test, y_pred)
print("Test Accuracy Score:", score)
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Feature importance
importances = dt_classifier.best_estimator_.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
print("\nFeature Importance:\n", feature_importance_df.sort_values(by='Importance', ascending=False))

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test_decoded,
    'Predicted': y_pred_decoded
})
predictions_df.to_csv('dt_predictions.csv', index=False)
print("\nPredictions saved to dt_predictions.csv")


# Function to test custom input
def test_custom_input(model, scaler, categorical_mappings, feature_names):
    print("\nEnter custom input for prediction (13 features):")
    print("Categorical feature encodings:")
    print("Gender: 0 (Female), 1 (Male)")
    print(
        "History, Patient, TakeMedication, BreathShortness, VisualChanges, NoseBleeding, ControlledDiet: 0 (No), 1 (Yes)")
    print("Severity: 0 (Mild), 1 (Moderate), 2 (Severe)")
    print("Whendiagnosed: 0 (1 - 5 Years), 1 (<1 Year), 2 (>5 Years)")

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

    print(f"\nPredicted Hypertension Stage: {custom_pred_decoded}")


# Run custom input test
test_custom_input(dt_classifier, scaler, categorical_mappings, X_train.columns)