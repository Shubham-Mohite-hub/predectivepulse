import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report
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

# Define numerical columns
numerical_cols = ['Age', 'Systolic', 'Diastolic']

# Initialize scaler and discretizer
scaler = StandardScaler()
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# Prepare scaled data
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Prepare discretized data for MultinomialNB
X_train_discrete = X_train.copy()
X_test_discrete = X_test.copy()
X_train_discrete[numerical_cols] = discretizer.fit_transform(X_train[numerical_cols])
X_test_discrete[numerical_cols] = discretizer.transform(X_test[numerical_cols])

# Apply SMOTE for all models
smote = SMOTE(random_state=42)
X_train_scaled_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
X_train_discrete_res, y_train_discrete_res = smote.fit_resample(X_train_discrete, y_train)

# Define models
models = {
    'Logistic_Regression_L1': LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42, class_weight='balanced'),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gaussian_NB': GaussianNB(),
    'Multinomial_NB': MultinomialNB(class_prior=(pd.Series(y_train).value_counts().sort_index() / len(y_train)).values)
}

# Function to train and evaluate a model
def train_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_decoded = pd.Series(y_test).map(categorical_mappings['Stages'])
    y_pred_decoded = pd.Series(y_pred).map(categorical_mappings['Stages'])
    score = accuracy_score(y_test, y_pred)
    print(f"\nResults for {name}:")
    print("Test Accuracy Score:", score)
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))
    predictions_df = pd.DataFrame({'Actual': y_test_decoded, 'Predicted': y_pred_decoded})
    predictions_df.to_csv(f'{name.lower()}_predictions.csv', index=False)
    print(f"Predictions saved to {name.lower()}_predictions.csv")
    return score, model

# Train and evaluate all models
results = {}
trained_models = {}
for name, model in models.items():
    if name == 'Multinomial_NB':
        score, trained_model = train_evaluate_model(name, model, X_train_discrete_res, y_train_discrete_res, X_test_discrete, y_test)
    else:
        score, trained_model = train_evaluate_model(name, model, X_train_scaled_res, y_train_res, X_test_scaled, y_test)
    results[name] = score
    trained_models[name] = trained_model

# Print summary
print("\nModel Comparison:")
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(results_df.sort_values(by='Accuracy', ascending=False))

# Function to test custom input
def test_custom_input(models, scaler, discretizer, categorical_mappings, feature_names, numerical_cols):
    print("\nSelect a model for custom input prediction:")
    for i, name in enumerate(models.keys(), 1):
        print(f"{i}. {name}")
    while True:
        try:
            choice = int(input("Enter model number: ")) - 1
            model_name = list(models.keys())[choice]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Try again.")

    model = models[model_name]
    print(f"\nEnter custom input for prediction (13 features) using {model_name}:")
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
                if model_name == 'Multinomial_NB' and value < 0:
                    print(f"{feature} must be non-negative.")
                    continue
                custom_input.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")

    custom_df = pd.DataFrame([custom_input], columns=feature_names)
    if model_name == 'Multinomial_NB':
        custom_df_processed = custom_df.copy()
        custom_df_processed[numerical_cols] = discretizer.transform(custom_df[numerical_cols])
    elif model_name in ['Logistic_Regression_L1', 'Gaussian_NB']:
        custom_df_processed = custom_df.copy()
        custom_df_processed[numerical_cols] = scaler.transform(custom_df[numerical_cols])
    else:
        custom_df_processed = custom_df

    custom_pred = model.predict(custom_df_processed)
    custom_pred_decoded = categorical_mappings['Stages'][custom_pred[0]]
    print(f"\nPredicted Hypertension Stage with {model_name}: {custom_pred_decoded}")

# Run custom input test
test_custom_input(trained_models, scaler, discretizer, categorical_mappings, X_train.columns, numerical_cols)