import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
data = pd.read_csv("blood_pressure_data.csv")

# Data Preprocessing
data = data.dropna()  # Drop missing values
data = data[(data['SystolicBP'] > 80) & (data['SystolicBP'] < 200)]  # Remove outliers
data = data[(data['DiastolicBP'] > 50) & (data['DiastolicBP'] < 130)]

# Feature Engineering
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 30, 50, 70, 100], labels=['0-30', '30-50', '50-70', '70+'])

# Exploratory Data Analysis (EDA)
# Summary statistics
print(data[['SystolicBP', 'DiastolicBP']].describe())

# Distribution Analysis
plt.figure(figsize=(12, 6))
sns.histplot(data['SystolicBP'], kde=True, color='blue', label='Systolic BP')
sns.histplot(data['DiastolicBP'], kde=True, color='orange', label='Diastolic BP')
plt.legend()
plt.title("Distribution of Blood Pressure")
plt.show()
