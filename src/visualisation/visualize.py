# visualize.py

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
