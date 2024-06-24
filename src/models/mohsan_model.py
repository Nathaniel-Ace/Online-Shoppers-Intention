import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Datenpfade
train_data_path = '/Users/mohsan/Documents/Studium/4.Semester/Data Science & AI/Online-Shoppers-Intention/data/processed/X_train.csv'
target_data_path = '/Users/mohsan/Documents/Studium/4.Semester/Data Science & AI/Online-Shoppers-Intention/data/processed/y_train.csv'

# Daten laden
X_train = pd.read_csv(train_data_path)
y_train = pd.read_csv(target_data_path)

# Zielvariable 'y' muss ein eindimensionaler Array sein
y_train = y_train['Revenue'].astype(int)

# Kategorialen Daten definieren
categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor', 'Weekend']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessor Pipeline erstellen
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough')

# Modell Pipeline erstellen
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000))])

# Modell trainieren
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_train)

# Modellleistung bewerten
accuracy = accuracy_score(y_train, y_pred)
report = classification_report(y_train, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
