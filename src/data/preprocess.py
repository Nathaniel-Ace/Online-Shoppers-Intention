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
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['VisitorType']  # Only encode 'VisitorType'
df_encoded = pd.DataFrame(encoder.fit_transform(df_cleaned[categorical_columns]), columns=encoder.get_feature_names_out())
df_encoded.index = df_cleaned.index  # Ensure indices are aligned by setting df_encoded index to match df_cleaned

# Join encoded columns back to the cleaned dataframe
df_cleaned = df_cleaned.drop(categorical_columns, axis=1).join(df_encoded)  # Drop original and join encoded columns
print("Shape of the dataset after encoding categorical variables:", df_cleaned.shape)
print("Sample data after encoding (specific columns):\n", df_cleaned[['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor']].head())


# Feature scaling
scaler = StandardScaler()
scaled_columns = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']  # Example of columns to scale
df_cleaned[scaled_columns] = scaler.fit_transform(df_cleaned[scaled_columns])
print("Sample data after scaling:\n", df_cleaned[scaled_columns].head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_cleaned.drop('Revenue', axis=1), df_cleaned['Revenue'], test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Save the training data
# X_train.to_csv('../../data/processed/X_train.csv', index=False)
# y_train.to_csv('../../data/processed/y_train.csv', index=False)

# Save the testing data
# X_test.to_csv('../../data/processed/X_test.csv', index=False)
# y_test.to_csv('../../data/processed/y_test.csv', index=False)

# Save the cleaned and preprocessed dataset (optional)
# df_cleaned.to_csv('../../data/processed/cleaned_online_shoppers_intention.csv', index=False)
