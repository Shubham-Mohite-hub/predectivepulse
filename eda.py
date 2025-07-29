import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Added for directory creation

# Ensure the 'eda_images' folder exists
if not os.path.exists('eda_images'):
    os.makedirs('eda_images')

# Print available Matplotlib styles for debugging
print("Available Matplotlib styles:", plt.style.available)

# Set style for visualizations
try:
    plt.style.use('seaborn-v0_8')  # Use a valid Seaborn style
except OSError:
    print("Warning: 'seaborn-v0_8' style not found, using 'default' style")
    plt.style.use('default')  # Fallback to default Matplotlib style
sns.set_palette("husl")

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Combine train and test for full dataset EDA
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)
data = pd.concat([X, y], axis=1)

# Define mappings for decoding categorical variables
categorical_mappings = {
    'Gender': {0: 'Female', 1: 'Male'},
    'History': {0: 'No', 1: 'Yes'},
    'Patient': {0: 'No', 1: 'Yes'},
    'TakeMedication': {0: 'No', 1: 'Yes'},
    'Severity': {0: 'Mild', 1: 'Moderate', 2: 'Severe'},
    'BreathShortness': {0: 'No', 1: 'Yes'},
    'VisualChanges': {0: 'No', 1: 'Yes'},
    'NoseBleeding': {0: 'No', 1: 'Yes'},
    'Whendiagnoused': {0: '1 - 5 Years', 1: '<1 Year', 2: '>5 Years'},
    'ControlledDiet': {0: 'No', 1: 'Yes'},
    'Stages': {0: 'HYPERTENSION (Stage-1)', 1: 'HYPERTENSION (Stage-2)',
               2: 'HYPERTENSIVE CRISIS', 3: 'NORMAL'}
}

# Decode categorical columns for EDA
data_decoded = data.copy()
for column, mapping in categorical_mappings.items():
    data_decoded[column] = data_decoded[column].map(mapping)

# 1. Descriptive Analysis
print("\nDescriptive Analysis for Continuous Features:")
print(data_decoded[['Age', 'Systolic', 'Diastolic']].describe())

print("\nDescriptive Analysis for Categorical Features:")
for col in categorical_mappings.keys():
    print(f"\n{col}:")
    print(f"Unique values: {data_decoded[col].unique()}")
    print(f"Top value: {data_decoded[col].mode()[0]}")
    print(f"Frequency of top value: {data_decoded[col].value_counts().iloc[0]}")
    print(f"Value counts:\n{data_decoded[col].value_counts()}")

# 2. Univariate Analysis
# Numerical Features (using histplot)
plt.figure(figsize=(15, 5))
for i, col in enumerate(['Age', 'Systolic', 'Diastolic'], 1):
    plt.subplot(1, 3, i)
    sns.histplot(data_decoded[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_images/univariate_numerical.png')
plt.close()

# Categorical Features (using countplot)
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_mappings.keys(), 1):
    plt.subplot(4, 3, i)
    sns.countplot(data=data_decoded, x=col)
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_images/univariate_categorical.png')
plt.close()

# 3. Gender Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
gender_counts = data_decoded['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.savefig('eda_images/gender_pie_chart.png')
plt.close()

# 4. Bivariate Analysis
# Numerical vs Numerical (Systolic vs Diastolic)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data_decoded, x='Systolic', y='Diastolic', hue='Stages')
plt.title('Systolic vs Diastolic by Stages')
plt.savefig('eda_images/bivariate_systolic_diastolic.png')
plt.close()

# Numerical vs Categorical (Age vs Stages)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_decoded, x='Stages', y='Age')
plt.title('Age vs Stages')
plt.xticks(rotation=45, ha='right')
plt.savefig('eda_images/bivariate_age_stages.png')
plt.close()

# Categorical vs Categorical (ControlledDiet vs Stages)
plt.figure(figsize=(8, 6))
sns.countplot(data=data_decoded, x='ControlledDiet', hue='Stages')
plt.title('ControlledDiet vs Stages')
plt.savefig('eda_images/bivariate_controlleddiet_stages.png')
plt.close()

# 5. Multivariate Analysis (Pairplot for Numerical Features)
sns.pairplot(data_decoded, vars=['Age', 'Systolic', 'Diastolic'], hue='Stages')
plt.savefig('eda_images/multivariate_pairplot.png')
plt.close()

# 6. Correlation Analysis (Numerical Features)
plt.figure(figsize=(8, 6))
correlation_matrix = data[['Age', 'Systolic', 'Diastolic']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('eda_images/correlation_matrix.png')
plt.close()

print("\nEDA completed. Visualizations saved as PNG files in eda_images folder.")