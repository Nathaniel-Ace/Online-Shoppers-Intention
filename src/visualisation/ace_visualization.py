import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('../../data/raw/online_shoppers_intention.csv')

# Initial shape
initial_shape = df.shape

# Display missing values count
missing_values = df.isnull().sum()

# Removing Duplicates
df_cleaned = df.drop_duplicates()
cleaned_shape = df_cleaned.shape

# Encoding Categorical Variables
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['VisitorType', 'Month']
df_encoded = pd.DataFrame(encoder.fit_transform(df_cleaned[categorical_columns]), columns=encoder.get_feature_names_out())
df_encoded.index = df_cleaned.index
df_cleaned = df_cleaned.drop(categorical_columns, axis=1).join(df_encoded)

# Feature Scaling
scaler = StandardScaler()
scaled_columns = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
df_cleaned[scaled_columns] = scaler.fit_transform(df_cleaned[scaled_columns])

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(df_cleaned.drop('Revenue', axis=1), df_cleaned['Revenue'], test_size=0.2, random_state=42)

# Plotting
# Missing Values Bar Chart
missing_values.plot(kind='bar', title='Missing Values Count')
plt.show()

# VisitorType Distribution Before Encoding
df['VisitorType'].value_counts().plot(kind='bar', title='VisitorType Distribution Before Encoding')
plt.show()

# VisitorType Distribution After Encoding
df_encoded[['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor']].sum().plot(kind='bar', title='VisitorType Distribution After Encoding')
plt.show()

# Feature Scaling Box Plot
df[scaled_columns].plot(kind='box', title='Feature Distribution Before Scaling')
plt.show()
df_cleaned[scaled_columns].plot(kind='box', title='Feature Distribution After Scaling')
plt.show()

# Pie Chart for Train-Test Split
sizes = [X_train.shape[0], X_test.shape[0]]
labels = ['Training Data', 'Testing Data']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# 1. Summary Statistics
print("Summary Statistics")
print(df.describe())

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
# Select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 3. Pairplot
# Select a subset of features for the pairplot to avoid overloading the plot
selected_features = ['Administrative', 'Informational', 'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues', 'Revenue']
sns.pairplot(df[selected_features], hue='Revenue', palette='viridis')
plt.title('Pairplot of Selected Features')
plt.show()

# 4. Histograms
df.hist(bins=20, figsize=(14, 10), color='steelblue', edgecolor='black', linewidth=1.2)
plt.suptitle('Histograms of Numerical Features')
plt.show()

# 5. Boxplots
plt.figure(figsize=(14, 10))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']), orient='h', palette='Set2')
plt.title('Boxplots of Numerical Features')
plt.show()

# 6. Dimensionality Reduction using PCA
# Handle categorical variables before PCA
df_encoded = pd.get_dummies(df, drop_first=True)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_encoded.select_dtypes(include=['float64', 'int64']))

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['Revenue'] = df['Revenue']

# Scatter plot of PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Revenue', data=pca_df, palette='viridis')
plt.title('PCA of Online Shoppers Intention Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Revenue', loc='upper right')
plt.show()
