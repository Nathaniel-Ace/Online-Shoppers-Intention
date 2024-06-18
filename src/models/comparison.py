import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Load the cleaned dataset
df_cleaned = pd.read_csv('../../data/processed/2cleaned_online_shoppers_intention.csv')

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

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Improved Logistic Regression': ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=10000, class_weight='balanced'))
    ]),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Splitting the data into a smaller training set and a validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a list to store performance metrics
metrics_list = []

# Train and evaluate each model
for name, model in models.items():
    if name == 'Improved Logistic Regression':
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_val)
    else:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train_split, y_train_split)
        y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    metrics_list.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)

# Print the performance metrics table
print(metrics_df)

# Plot performance metrics
metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(14, 8))
plt.title('Comparison of Model Performance')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.show()

# Highlight the best-performing model
best_model = metrics_df['F1-Score'].idxmax()
print(f'The best-performing model is: {best_model}')

# Print detailed classification report for the best model
best_model_pipeline = models[best_model] if best_model != 'Improved Logistic Regression' else ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=10000, class_weight='balanced'))
])
best_model_pipeline.fit(X_train_split, y_train_split)
y_pred_best = best_model_pipeline.predict(X_val)
print(classification_report(y_val, y_pred_best))
