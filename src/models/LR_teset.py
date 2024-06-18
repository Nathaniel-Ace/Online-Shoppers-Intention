import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

print("logistic_model.joblib")

# Load the test datasets
X_test = pd.read_csv('../../data/processed/2X_test.csv')
y_test = pd.read_csv('../../data/processed/2y_test.csv')

# Load the trained model
model = load('logistic_model.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Print the evaluation results for the test set
print("Test Set Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Load the cleaned dataset
df_cleaned = pd.read_csv('../../data/processed/2cleaned_online_shoppers_intention.csv')

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

# Confusion Matrix Visualization
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_cleaned, annot=True, fmt='d', cmap='Reds', annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Bar Chart for Classification Report
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

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.legend(loc="lower right")
plt.show()
