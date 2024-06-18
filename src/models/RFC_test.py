import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score
from joblib import load
import seaborn as sns

print("RFC.joblib")

# Load the model
model = load('RFC.joblib')

# Load the test data
X_test = pd.read_csv('../../data/processed/X_test.csv')
y_test = pd.read_csv('../../data/processed/y_test.csv')

# Flatten the y_test DataFrame to Series
y_test = y_test['Revenue']

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print("Evaluation results for the test data:")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Load the cleaned dataset
df_cleaned = pd.read_csv('../../data/processed/cleaned_online_shoppers_intention.csv')

# Split the cleaned dataset into features and target
X_cleaned = df_cleaned.drop('Revenue', axis=1)
y_cleaned = df_cleaned['Revenue']

# Predict on the cleaned data
y_pred_cleaned = model.predict(X_cleaned)

# Evaluate the model on the cleaned data
accuracy_cleaned = accuracy_score(y_cleaned, y_pred_cleaned)
conf_matrix_cleaned = confusion_matrix(y_cleaned, y_pred_cleaned)
class_report_cleaned = classification_report(y_cleaned, y_pred_cleaned)

# Print the evaluation results for the cleaned data
print("Evaluation results for the cleaned data:")
print(f"Accuracy: {accuracy_cleaned}")
print("Confusion Matrix:")
print(conf_matrix_cleaned)
print("Classification Report:")
print(class_report_cleaned)

# Plot Confusion Matrix using seaborn heatmap with larger fonts
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            annot_kws={"size": 14},  # Increase the font size of the annotations
            cbar_kws={"label": "Count"})  # Add label to color bar
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_cleaned, annot=True, fmt='d', cmap='Reds',
            annot_kws={"size": 14},  # Increase the font size of the annotations
            cbar_kws={"label": "Count"})  # Add label to color bar
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Performance Metrics (Example: Accuracy, Precision, Recall, F1-Score)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Bar Chart for Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]

plt.bar(metrics, scores, color=['blue', 'orange', 'green', 'red'])
plt.title('Model Performance Metrics')
plt.ylim(0, 1)
plt.show()
