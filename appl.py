import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('patient_data.csv')

# Rename 'C' column to 'Gender'
df = df.rename(columns={'C': 'Gender'})

# Function to preprocess blood pressure ranges (Systolic, Diastolic) and Age ranges
def process_range(value, is_age=False):
    if is_age:
        if value == '65+':
            return 65.0  # Assume 65 for '65+'
        elif '-' in value:
            low, high = map(int, value.split('-'))
            return (low + high) / 2.0
    else:
        if value in ['130+', '100+']:
            return float(value[:-1])  # Remove '+' and convert to float
        elif '-' in value:
            low, high = map(int, value.split('-'))
            return (low + high) / 2.0
    return float(value)

# 1. Handle typos and inconsistencies
# Correct 'Severity' typo
df['Severity'] = df['Severity'].replace('Sever', 'Severe')

# Correct 'Stages' typos
df['Stages'] = df['Stages'].replace('HYPERTENSIVE CRISI', 'HYPERTENSIVE CRISIS')
df['Stages'] = df['Stages'].replace('HYPERTENSION (Stage-2).', 'HYPERTENSION (Stage-2)')

# Standardize categorical columns: remove extra spaces and convert to title case
categorical_columns = ['Gender', 'History', 'Patient', 'TakeMedication',
                      'Severity', 'BreathShortness', 'VisualChanges',
                      'NoseBleeding', 'Whendiagnoused', 'ControlledDiet', 'Stages']
for column in categorical_columns:
    df[column] = df[column].str.strip().str.title()

# Print unique values in categorical columns before encoding
print("Unique values in categorical columns after standardization:")
for column in categorical_columns:
    print(f"{column}: {df[column].unique()}")

# 2. Process Age, Systolic, and Diastolic (no scaling)
df['Age'] = df['Age'].apply(lambda x: process_range(x, is_age=True))
df['Systolic'] = df['Systolic'].apply(process_range)
df['Diastolic'] = df['Diastolic'].apply(process_range)

# 3. Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    print(f"{column} LabelEncoder mapping:", list(zip(le.classes_, range(len(le.classes_)))))

# Verify ControlledDiet encoding explicitly
print("\nControlledDiet sample (first 10 rows):")
print(df[['ControlledDiet']].head(10))

# Print unique values in processed numerical columns
print("\nUnique values in numerical columns after preprocessing:")
print(f"Age (numerical): {df['Age'].unique()}")
print(f"Systolic (numerical): {df['Systolic'].unique()}")
print(f"Diastolic (numerical): {df['Diastolic'].unique()}")

# 4. Split features and target
X = df.drop('Stages', axis=1)
y = df['Stages']

# 5. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Decode y_test to original categories for verification
y_test_decoded = label_encoders['Stages'].inverse_transform(y_test)
print("\nDecoded y_test values:", y_test_decoded[:20])
print("Corresponding y_test encoded values:", y_test[:20].values)

# Print basic information about the preprocessed data
print("\nPreprocessed Data Info:")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\nFeature columns:", X.columns.tolist())