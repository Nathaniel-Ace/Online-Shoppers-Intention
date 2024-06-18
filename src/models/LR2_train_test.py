import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("logistic_model_improved.joblib")

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

# Create a pipeline with SMOTE and Logistic Regression with class weights
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=10000, class_weight='balanced'))
])

# Splitting the data into a smaller training set and a validation set
X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train_split, y_train_split)

# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Evaluate the model on the validation set
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred, output_dict=True)

# Save the model
dump(pipeline, 'logistic_model_improved.joblib')

# Print the evaluation results for the validation set
print("Validation Set Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_val, y_pred))

# Evaluate the model on the entire cleaned dataset
y_pred_cleaned = pipeline.predict(X)

accuracy_cleaned = accuracy_score(y, y_pred_cleaned)
conf_matrix_cleaned = confusion_matrix(y, y_pred_cleaned)
class_report_cleaned = classification_report(y, y_pred_cleaned, output_dict=True)

print("Evaluation results for the cleaned data:")
print(f"Accuracy: {accuracy_cleaned}")
print("Confusion Matrix:")
print(conf_matrix_cleaned)
print("Classification Report:")
print(class_report_cleaned)

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', annot_kws={"size": 14})
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Plot Confusion Matrix for Validation Set
plot_confusion_matrix(conf_matrix, title='Confusion Matrix - Validation Set')

# Plot Confusion Matrix for Cleaned Dataset
plot_confusion_matrix(conf_matrix_cleaned, title='Confusion Matrix - Cleaned Dataset')

# Function to plot classification report
def plot_classification_report(class_report, title='Classification Report'):
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(class_report.keys())

    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        scores = [class_report[c][metric] for c in classes if c != 'accuracy']
        plt.bar(classes[:-1], scores, color=['blue', 'orange'])
        plt.title(f'Class {metric.capitalize()}', fontsize=14)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Class', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
    plt.tight_layout()
    plt.show()

# Plot Classification Report for Validation Set
plot_classification_report(class_report, title='Classification Report - Validation Set')

# Plot Classification Report for Cleaned Dataset
plot_classification_report(class_report_cleaned, title='Classification Report - Cleaned Dataset')

# ROC Curve for Validation Set
y_prob = pipeline.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic - Validation Set', fontsize=16)
plt.legend(loc="lower right")
plt.show()

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
