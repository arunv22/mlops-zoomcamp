import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment_module2")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # Enable autologging
    mlflow.autolog()
    with mlflow.start_run():
        # Load data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        mlflow.set_tag("developer", "arun")
        mlflow.log_param("train-data-path", "/workspaces/mlops-zoomcamp/cohorts/2024/data/green_tripdata_2023-01.parquet")
        mlflow.log_param("valid-data-path", "/workspaces/mlops-zoomcamp/cohorts/2024/data/green_tripdata_2023-02.parquet")
        mlflow.log_param("test-data-path", "/workspaces/mlops-zoomcamp/cohorts/2024/data/green_tripdata_2023-03.parquet")

        # Train model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        # Predict on validation set
        y_pred = rf.predict(X_val)

        # Calculate RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Log RMSE metric
        mlflow.log_metric("rmse", rmse)
        
        # Save the model to a file
        model_dir = "models"
        model_filename = "model_rf.bin"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_filename)
        with open(model_path, "wb") as f_out:
            pickle.dump(rf, f_out)
        # Log the model file as an artifact
        print(f"local_path is {model_path}")
        print("***************************************")
        mlflow.log_artifact(local_path=model_path, artifact_path="models")
        print("***************************************")

if __name__ == '__main__':
    run_train()
