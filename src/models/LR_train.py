import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
X_train = pd.read_csv('../../data/processed/2X_train.csv')
y_train = pd.read_csv('../../data/processed/2y_train.csv')
X_test = pd.read_csv('../../data/processed/2X_test.csv')
y_test = pd.read_csv('../../data/processed/2y_test.csv')

# Load the cleaned dataset
df_cleaned = pd.read_csv('../../data/processed/2cleaned_online_shoppers_intention.csv')

# Check for missing values in the dataset
missing_values = df_cleaned.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Compute the correlation matrix
correlation_matrix = df_cleaned.corr()

# Threshold for low correlation
low_corr_threshold = 0.05

# Identify features with low correlation to 'Revenue'
low_corr_features = correlation_matrix.index[correlation_matrix['Revenue'].abs() < low_corr_threshold].tolist()

# Drop low correlation features
X_filtered = df_cleaned.drop(columns=low_corr_features)

# Ensure target variable 'Revenue' is in y and remove it from X
X = X_filtered.drop('Revenue', axis=1)
y = X_filtered['Revenue']

# Identify categorical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns.tolist()),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Creating the final pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=10000))])

# Splitting the data into a smaller training set and a validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train_split, y_train_split)

# Predicting on the validation set
y_pred = pipeline.predict(X_val)

# Calculating accuracy and other metrics
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

# Save the model
# dump(pipeline, 'logistic_model.joblib')

# Displaying the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Displaying the dropped features
print("Dropped Features:", low_corr_features)

# Learning Curve Visualization
train_sizes, train_scores, test_scores = learning_curve(pipeline, X_train_split, y_train_split, cv=5, scoring='accuracy', n_jobs=-1)

plt.figure(figsize=(10, 7))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.xlabel('Training Size', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Learning Curve', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Cross-Validation Scores Visualization
cv_scores = cross_val_score(pipeline, X_train_split, y_train_split, cv=5, scoring='accuracy')
plt.figure(figsize=(10, 7))
plt.bar(range(1, 6), cv_scores, color='blue')
plt.xlabel('Cross-Validation Fold', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Cross-Validation Scores', fontsize=16)
plt.ylim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
