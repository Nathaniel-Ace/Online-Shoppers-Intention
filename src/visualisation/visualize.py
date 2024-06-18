import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_matrix(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Convert boolean columns to integers if necessary
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def plot_revenue_pie_chart(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Ensure Revenue is treated as a categorical variable
    revenue_counts = df['Revenue'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(revenue_counts, labels=revenue_counts.index, autopct=lambda p: '{:.1f}%'.format(p), startangle=140, colors=['skyblue', 'salmon'], textprops={'fontsize': 14})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Plot the correlation matrix heatmap
plot_correlation_matrix('../../data/raw/online_shoppers_intention.csv')

# Plot the revenue pie chart
plot_revenue_pie_chart('../../data/raw/online_shoppers_intention.csv')
