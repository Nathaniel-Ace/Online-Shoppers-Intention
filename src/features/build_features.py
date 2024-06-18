import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the cleaned dataset
df_cleaned = pd.read_csv('../../data/raw/online_shoppers_intention.csv')

# Interaction Features
df_cleaned['ProductRelated_Duration'] = df_cleaned['ProductRelated'] * df_cleaned['ProductRelated_Duration']

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df_cleaned[['Administrative', 'Informational', 'ProductRelated']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Administrative', 'Informational', 'ProductRelated']))

# Append polynomial features to the original dataframe
df_cleaned = pd.concat([df_cleaned, poly_features_df], axis=1)

# Log Transformation
df_cleaned['log_PageValues'] = np.log1p(df_cleaned['PageValues'])

# Binning
df_cleaned['BounceRates_bin'] = pd.cut(df_cleaned['BounceRates'], bins=5, labels=False)

# Scaling
scaler = StandardScaler()
df_cleaned[['Administrative', 'Informational', 'ProductRelated', 'PageValues']] = scaler.fit_transform(df_cleaned[['Administrative', 'Informational', 'ProductRelated', 'PageValues']])

# Encoding Categorical Variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_month = encoder.fit_transform(df_cleaned[['Month']])
encoded_month_df = pd.DataFrame(encoded_month, columns=encoder.get_feature_names_out(['Month']))

# Append encoded features to the original dataframe
df_cleaned = pd.concat([df_cleaned, encoded_month_df], axis=1)

# Drop original categorical columns
df_cleaned.drop(['Month'], axis=1, inplace=True)

# Save the engineered dataset
df_cleaned.to_csv('../../data/processed/engineered_online_shoppers_intention.csv', index=False)

print("Feature engineering applied and dataset saved.")
