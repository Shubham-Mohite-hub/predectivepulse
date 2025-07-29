import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load preprocessed data
X_train = pd.read_csv('../X_train.csv')
X_test = pd.read_csv('../X_test.csv')
y_train = pd.read_csv('../y_train.csv')
y_test = pd.read_csv('../y_test.csv')

# Ensure y_train and y_test are 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Print training data statistics
print("\nTraining Data Statistics:")
print(X_train.describe())
print("\nClass Distribution:")
categorical_mappings = {
    'Stages': {0: 'HYPERTENSION (Stage-1)', 1: 'HYPERTENSION (Stage-2)',
               2: 'HYPERTENSIVE CRISIS', 3: 'NORMAL'}
}
print(pd.Series(y_train).value_counts().map(categorical_mappings['Stages']))

# Discretize continuous features
numerical_cols = ['Age', 'Systolic', 'Diastolic']
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_train_discrete = X_train.copy()
X_test_discrete = X_test.copy()
X_train_discrete[numerical_cols] = discretizer.fit_transform(X_train[numerical_cols])
X_test_discrete[numerical_cols] = discretizer.transform(X_test[numerical_cols])

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_discrete, y_train)


# Function to train and evaluate Multinomial Naive Bayes
def nb_model(X_train, y_train, X_test, y_test):
    # Initialize Multinomial Naive Bayes model with class priors
    class_counts = pd.Series(y_train).value_counts()
    total = class_counts.sum()
    priors = class_counts.sort_index() / total
    nb = MultinomialNB(class_prior=priors.values)

    # Train the model
    nb.fit(X_train, y_train)

    # Predict on test data
    y_pred = nb.predict(X_test)

    # Decode predictions and actuals for classification report
    y_test_decoded = pd.Series(y_test).map(categorical_mappings['Stages'])
    y_pred_decoded = pd.Series(y_pred).map(categorical_mappings['Stages'])

    # Evaluate model
    score = accuracy_score(y_test, y_pred)
    print("Test Accuracy Score:", score)
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_test_decoded,
        'Predicted': y_pred_decoded
    })
    predictions_df.to_csv('nb_predictions.csv', index=False)
    print("\nPredictions saved to nb_predictions.csv")

    return nb, y_pred


# Train and evaluate the model
nb_classifier, y_pred = nb_model(X_train_res, y_train_res, X_test_discrete, y_test)


# Function to test custom input
def test_custom_input(model, discretizer, categorical_mappings, feature_names, numerical_cols):
    print("\nEnter custom input for prediction (13 features):")
    print("Categorical feature encodings:")
    print("Gender: 0 (Female), 1 (Male)")
    print(
        "History, Patient, TakeMedication, BreathShortness, VisualChanges, NoseBleeding, ControlledDiet: 0 (No), 1 (Yes)")
    print("Severity: 0 (Mild), 1 (Moderate), 2 (Severe)")
    print("Whendiagnosed: 0 (1 - 5 Years), 1 (<1 Year), 2 (>5 Years)")
    print("Valid ranges: Age (18-100), Systolic (90-250), Diastolic (50-150)")

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
                # Validate numerical features
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
                # Ensure non-negative for MultinomialNB
                if value < 0:
                    print(f"{feature} must be non-negative.")
                    continue
                custom_input.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")

    # Create DataFrame for input using training feature names
    custom_df = pd.DataFrame([custom_input], columns=feature_names)

    # Discretize numerical features
    custom_df_discrete = custom_df.copy()
    custom_df_discrete[numerical_cols] = discretizer.transform(custom_df[numerical_cols])

    # Predict
    custom_pred = model.predict(custom_df_discrete)
    custom_pred_decoded = categorical_mappings['Stages'][custom_pred[0]]

    print(f"\nPredicted Hypertension Stage: {custom_pred_decoded}")


# Run custom input test
test_custom_input(nb_classifier, discretizer, categorical_mappings, X_train.columns, numerical_cols)