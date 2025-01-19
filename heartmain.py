Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
url = "/Users/nishashukla/Downloads/heart.csv"

# Column names for the dataset (since the dataset doesn't include headers)
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
    "oldpeak", "slope", "ca", "thal", "target"
]

# Lnames=column_names)
data = pd.read_csv(url, names=column_names, header=0) 
# Display first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Step 2: Data Preprocessing
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Handle missing values with SimpleImputer (replace missing with mean for numerical columns)
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Step 3: Exploratory Data Analysis (EDA)
# Visualizing the correlation matrix to understand the relationships between features
plt.figure(figsize=(12, 8))
sns.heatmap(data_imputed.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Visualizing the distribution of the target variable ('target')
sns.countplot(x='target', data=data_imputed)
plt.title('Heart Disease Distribution')
plt.show()

# Step 4: Feature and Target Variables
X = data_imputed.drop('target', axis=1)  # Features
y = data_imputed['target']  # Target variable (heart disease: 1, no disease: 0)

# Step 5: Splitting the Data into Training and Testing Sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Data Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: ", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))