import pandas as pd
import matplotlib.pyplot as plt

# Define the performance metrics for the cleaned data
metrics_data = {
    'Model': ['RFC.joblib', 'logistic_model.joblib', 'logistic_model_improved.joblib'],
    'Accuracy': [0.9807, 0.8837, 0.8619],
    'Precision (True)': [0.96, 0.75, 0.54],
    'Recall (True)': [0.92, 0.38, 0.76],
    'F1-Score (True)': [0.94, 0.51, 0.63]
}

# Create a DataFrame from the metrics data
metrics_df = pd.DataFrame(metrics_data)

# Print the performance metrics table
print(metrics_df)

# Plot performance metrics
metrics_df.set_index('Model', inplace=True)
ax = metrics_df.plot(kind='bar', figsize=(14, 8))
plt.title('Model Performance on Cleaned Data')
plt.ylabel('Score')
plt.xticks(rotation=0)  # Set x-axis labels to be horizontal
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Highlight the best-performing model
best_model = metrics_df['F1-Score (True)'].idxmax()
print(f'The best-performing model is: {best_model}')

