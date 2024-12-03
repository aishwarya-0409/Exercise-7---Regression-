'''linear regression '''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('./google-play-store-apps/googleplaystore.csv')

# Preprocess the data
def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()

    # Convert 'Installs' to numeric
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)

    # Convert 'Price' to numeric
    df['Price'] = df['Price'].str.replace('$', '').astype(float)

    # Encode 'Category' as numeric
    df['Category'] = df['Category'].astype('category').cat.codes

    # Use numeric features only
    numeric_cols = ['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Price']
    df = df[numeric_cols]

    return df
df = preprocess_data(df)

# Define models with feature combinations
models = [
    (['Reviews'], 'Rating'),
    (['Reviews', 'Installs'], 'Rating'),
    (['Reviews', 'Installs', 'Price'], 'Rating'),
    (['Category', 'Reviews'], 'Rating'),
    (['Category', 'Reviews', 'Price'], 'Rating')
]

# Train and evaluate models
for i, (features, target) in enumerate(models, 1):
    # Split the data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print scores
    print(f"Model {i} - Features: {features}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print()

    # Plot actual vs. predicted values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
    plt.title(f'Model {i} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

