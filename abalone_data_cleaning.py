import pandas as pd

# Load the dataset
file_path = './Abalone_Data_Set_[2874].xlsx'
abalone_data = pd.read_excel(file_path)

# Imputation for missing values
# Impute missing numerical values with the mean of the respective columns
abalone_data['Length(mm)'].fillna(abalone_data['Length(mm)'].mean(), inplace=True)
abalone_data['Diameter(mm)'].fillna(abalone_data['Diameter(mm)'].mean(), inplace=True)
abalone_data['Height(mm)'].fillna(abalone_data['Height(mm)'].mean(), inplace=True)
abalone_data['WholeWeight(g)'].fillna(abalone_data['WholeWeight(g)'].mean(), inplace=True)
abalone_data['ShuckedWeight(g)'].fillna(abalone_data['ShuckedWeight(g)'].mean(), inplace=True)
abalone_data['SellWeight(g)'].fillna(abalone_data['SellWeight(g)'].mean(), inplace=True)

# Handling zero values in critical columns by replacing them with the mean
abalone_data['Length(mm)'] = abalone_data['Length(mm)'].replace(0, abalone_data['Length(mm)'].mean())
abalone_data['Diameter(mm)'] = abalone_data['Diameter(mm)'].replace(0, abalone_data['Diameter(mm)'].mean())
abalone_data['Height(mm)'] = abalone_data['Height(mm)'].replace(0, abalone_data['Height(mm)'].mean())
abalone_data['WholeWeight(g)'] = abalone_data['WholeWeight(g)'].replace(0, abalone_data['WholeWeight(g)'].mean())
abalone_data['ShuckedWeight(g)'] = abalone_data['ShuckedWeight(g)'].replace(0, abalone_data['ShuckedWeight(g)'].mean())
abalone_data['SellWeight(g)'] = abalone_data['SellWeight(g)'].replace(0, abalone_data['SellWeight(g)'].mean())

# Verify no missing zero values remain
missing_data_after = abalone_data.isnull().sum()
print("Missing data after cleaning:\n", missing_data_after)

# Save cleaned dataset to new file
abalone_data.to_csv('cleaned_abalone_data_2.csv', index=False)
