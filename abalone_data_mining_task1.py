import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './Abalone_Data_Set_[2874].xlsx'  # Update the file path accordingly
abalone_data = pd.read_excel(file_path)

# Initial Exploration - Display first few rows of the dataset
print("First five rows of the dataset:")
print(abalone_data.head())

# Check for missing data
missing_data = abalone_data.isnull().sum()
print("\nMissing data in each column:")
print(missing_data)

# Check data types
data_types = abalone_data.dtypes
print("\nData types of each column:")
print(data_types)

# Descriptive statistics
summary_stats = abalone_data.describe()
print("\nSummary statistics of the dataset:")
print(summary_stats)

# Visualization of data distributions (histograms)
abalone_data.hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograms of Abalone Dataset Features')
plt.show()

# Exclude non-numeric columns (like 'Sex') for the correlation matrix
numeric_columns = abalone_data.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Matrix of Abalone Dataset (Numeric Columns Only)')
plt.show()


# Saved cleaned data
abalone_data.to_csv('cleaned_abalone_data_1.csv', index=False)
