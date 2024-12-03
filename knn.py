'''KNN'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./google-play-store-apps/googleplaystore.csv')


# Data Preprocessing
def preprocess_data(df):
    # Drop rows with missing 'Rating' as we need this for classification
    df = df.dropna(subset=['Rating'])

    # Handle missing values in other columns (e.g., Size, Reviews, Price) by filling with median
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df['Size'] = df['Size'].fillna(df['Size'].median())  # Fill missing values with median

    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median())  # Fill missing values with median

    df['Price'] = df['Price'].replace('NaN', '0')  # Replace NaN in 'Price' with '0' if necessary
    df['Price'] = df['Price'].astype(str).str.replace('$', '').astype(float)
    df['Price'] = df['Price'].fillna(df['Price'].median())  # Fill missing values with median

    # Clean 'Installs' by removing commas and '+' and convert to int
    df['Installs'] = df['Installs'].astype(str).str.replace(',', '').str.replace('+', '').astype(int)

    # Encode 'Category' as numeric
    df['Category'] = df['Category'].astype('category').cat.codes

    # Create a binary target variable 'High_Rating' (1 for ratings >= 4, else 0)
    df['High_Rating'] = np.where(df['Rating'] >= 4, 1, 0)

    # Select relevant features for prediction
    features = ['Category', 'Reviews', 'Size', 'Installs', 'Price']

    return df[features], df['High_Rating']

# Preprocess data and split features (X) and target (y)
X, y = preprocess_data(df)

# Ensure there are no NaN values in X and y
print(f"Checking for NaN values in the features:\n{X.isna().sum()}")
print(f"Checking for NaN values in the target variable:\n{y.isna().sum()}")

# Drop any remaining rows with NaN values
X = X.dropna()
y = y[X.index]  # Keep target aligned with features after dropping NaNs

# Check if data is empty after dropping NaN values
print(f"Shape of X after dropping NaNs: {X.shape}")
print(f"Shape of y after dropping NaNs: {y.shape}")

# Split the dataset into training and testing sets (70% training, 30% testing)
if X.shape[0] > 0 and y.shape[0] > 0:  # Ensure that there are still data points left
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features (important for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the KNN classifier with k=5
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix using a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Rating', 'High Rating'], yticklabels=['Low Rating', 'High Rating'])
    plt.title('Confusion Matrix for KNN Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

else:
    print("Data is empty after dropping NaN values. Please check the dataset.")


