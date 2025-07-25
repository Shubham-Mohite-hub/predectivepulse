import pandas as pd

# Load the dataset
df = pd.read_csv('patient_data.csv')

# Rename column 'C' to 'Gender' if needed
df = df.rename(columns={'C': 'Gender'})

# Remove extra spaces from all string values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Fill any missing values using forward fill
df = df.ffill()

# Convert Age to number (handles '65+', '51 - 64', etc.)
def age_to_number(age):
    if isinstance(age, str):
        age = age.replace(' ', '')
        if '-' in age:
            start, end = age.split('-')
            return (int(start) + int(end)) // 2
        elif '+' in age:
            return int(age.replace('+', ''))
    try:
        return int(age)
    except:
        return None

df['Age'] = df['Age'].apply(age_to_number)

# Convert BP values to average if range (e.g. '120 - 140')
def bp_to_number(bp):
    if isinstance(bp, str):
        bp = bp.replace(' ', '')
        if '-' in bp:
            start, end = bp.split('-')
            return (int(start) + int(end)) // 2
    try:
        return int(bp)
    except:
        return None

df['Systolic'] = df['Systolic'].apply(bp_to_number)
df['Diastolic'] = df['Diastolic'].apply(bp_to_number)

# Drop unnecessary columns
df = df.drop(columns=['Patient', 'Whendiagnoused'], errors='ignore')

# Show dataset info and preview
df.info()
print(df.head())

# (Optional) Save to new CSV file
# df.to_csv('cleaned_patient_data_no_encoding.csv', index=False)
