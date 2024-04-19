import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../../data/raw/online_shoppers_intention.csv')
print("Initial shape of the dataset:", df.shape)

# Display missing values count
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Remove duplicate rows and explicitly create a copy to avoid SettingWithCopyWarning
df_cleaned = df.drop_duplicates().copy()
print("Shape of the dataset after removing duplicates:", df_cleaned.shape)

# Encoding categorical variables
encoder = OneHotEncoder(sparse_output=False)  # Correct setting to avoid sparse matrix issues
categorical_columns = ['Month', 'VisitorType']  # Example of categorical columns
df_encoded = pd.DataFrame(encoder.fit_transform(df_cleaned[categorical_columns]), columns=encoder.get_feature_names_out())
df_cleaned = df_cleaned.drop(categorical_columns, axis=1).join(df_encoded)  # Drop and join without inplace modification
print("Shape of the dataset after encoding categorical variables:", df_cleaned.shape)
print("Sample data after encoding:\n", df_cleaned.head())

# Feature scaling
scaler = StandardScaler()
scaled_columns = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']  # Example of columns to scale
df_cleaned[scaled_columns] = scaler.fit_transform(df_cleaned[scaled_columns])
print("Sample data after scaling:\n", df_cleaned[scaled_columns].head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_cleaned.drop('Revenue', axis=1), df_cleaned['Revenue'], test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Save the cleaned and preprocessed dataset (optional)
# df_cleaned.to_csv('../../data/processed/cleaned_online_shoppers_intention.csv', index=False)
