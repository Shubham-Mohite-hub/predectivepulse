import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('patient_data.csv')
pd.DataFrame()
df = df.rename(columns={'C': 'Gender'})
# Strip whitespace from all string columns
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.str.strip())

# Forward-fill missing values (alternative to fillna(method='ffill'))
df.ffill(inplace=True)

# Convert Age from ranges like "51-64" to midpoints
def convert_age_range(age):
    try:
        parts = age.replace(' ', '').split('-')
        return (int(parts[0]) + int(parts[1])) // 2
    except:
        return np.nan

df['Age'] = df['Age'].apply(convert_age_range).astype('Int64')

# Convert Systolic and Diastolic ranges to numeric midpoints
def convert_range_to_int(val):
    try:
        parts = val.replace(' ', '').split('-')
        return (int(parts[0]) + int(parts[1])) // 2
    except:
        return np.nan

df['Systolic'] = df['Systolic'].apply(convert_range_to_int).astype('Int64')
df['Diastolic'] = df['Diastolic'].apply(convert_range_to_int).astype('Int64')

# Encode binary columns: Yes/No -> 1/0
binary_cols = ['History', 'TakeMedication', 'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Label encode categorical columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])     # Female = 0, Male = 1 (assumed)
df['Stages'] = le.fit_transform(df['Stages'])     # Convert disease stage to numerical
df['Severity'] = le.fit_transform(df['Severity']) # Encode severity

# Drop irrelevant or unique columns
df.drop(columns=['Patient', 'Whendiagnoused'], inplace=True)

# Final data check
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nPreview of cleaned data:\n", df.head())

