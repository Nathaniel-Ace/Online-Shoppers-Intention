import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from joblib import dump
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
X_train = pd.read_csv('../../data/processed/X_train.csv')
y_train = pd.read_csv('../../data/processed/y_train.csv')

# Flatten the y_train DataFrame to Series
y_train = y_train['Revenue']

# Identify the categorical and numeric columns
categorical_features = ['Month']
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create a column transformer to handle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that first preprocesses the data and then fits the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Define the cross-validation strategy
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_results}")
print(f"Mean Accuracy: {np.mean(cv_results)}")
print(f"Standard Deviation: {np.std(cv_results)}")

# Fit the model on the entire training data
model.fit(X_train, y_train)

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Save the model
# dump(model, 'RFC.joblib')
