from azureml.core import Workspace, Experiment, Run
from train import load_data, train_model, evaluate_model, save_plot
from sklearn.model_selection import train_test_split  # Add this import
import os

# Define data and output directories
DATA_DIR = os.path.join("data", "housing.data")  # Path to the dataset
OUTPUT_DIR = "outputs"  # Directory to save outputs

# Load Azure ML workspace
ws = Workspace.from_config()

# Create an experiment
experiment = Experiment(workspace=ws, name="house-price-prediction")

# Start a run
with experiment.start_logging() as run:
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

    # Log metrics
    run.log("MSE", mse)
    run.log("R²", r2)

    # Save plot
    save_plot(y_test, model.predict(X_test), OUTPUT_DIR)

    # Upload outputs to Azure ML
    run.upload_folder(name="outputs", path=OUTPUT_DIR)

    # Complete the run
    run.complete()