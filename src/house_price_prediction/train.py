import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define data and output directories
DATA_DIR = os.path.join("data", "housing.data")  # Path to the dataset
OUTPUT_DIR = "outputs"  # Directory to save outputs

def load_data(data_path):
    """Load dataset from the housing.data file."""
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    return pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names)

def train_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def save_plot(y_test, y_pred, output_dir):
    """Save a plot of actual vs predicted values."""
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices (MEDV)")
    plt.ylabel("Predicted Prices (MEDV)")
    plt.title("Actual vs Predicted Prices")
    plt.savefig(os.path.join(output_dir, "results.png"))

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data(DATA_DIR)

    # Preprocess data
    X = df.drop("MEDV", axis=1)  # Features
    y = df["MEDV"]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"MSE: {mse}, R²: {r2}")

    # Save plot
    save_plot(y_test, model.predict(X_test), OUTPUT_DIR)

if __name__ == "__main__":
    main()