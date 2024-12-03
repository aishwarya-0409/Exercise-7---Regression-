''' logistic regression '''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./google-play-store-apps/googleplaystore.csv')

# Preprocessing the data
def preprocess_data(df):
    # Drop rows with missing 'Rating' as it's crucial for our target variable
    df = df.dropna(subset=['Rating'])

    # Ensure 'Installs' is treated as string before processing
    df['Installs'] = df['Installs'].astype(str)
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(float)

    # Remove '$' and convert 'Price' to numeric
    df['Price'] = df['Price'].astype(str).str.replace('$', '').astype(float)

    # Handle missing or non-numeric values in 'Size' by converting to numeric, errors become NaN
    df['Size'] = df['Size'].replace('Varies with device', np.nan)  # Replace 'Varies with device' with NaN
    df['Size'] = df['Size'].astype(str).str.replace('M', 'e6').str.replace('k', 'e3')  # Convert size to standard numbers
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')  # Coerce invalid entries to NaN
    df = df.dropna(subset=['Size'])  # Drop rows where 'Size' could not be converted

    # Encode 'Category' as numeric using LabelEncoder
    df['Category'] = df['Category'].astype('category').cat.codes

    # Create a binary classification for 'Rating' (High vs Low Rating)
    df['High_Rating'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

    # Select relevant features
    X = df[['Category', 'Reviews', 'Size', 'Installs', 'Price']]
    y = df['High_Rating']

    return X, y

# Preprocess the data
X, y = preprocess_data(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{cm}")

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Rating', 'High Rating'], yticklabels=['Low Rating', 'High Rating'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
