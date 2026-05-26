import os
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocess import load_data, preprocess_data, split_data


mlflow.set_experiment("mlops-project")


def train():
    # Load dataset
    df = load_data("../data/data.csv")

    # Preprocess
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model
    model = RandomForestClassifier(n_estimators=100)

    with mlflow.start_run():

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print("Accuracy:", acc)
        print("F1 Score:", f1)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save model
        os.makedirs("../models", exist_ok=True)

        joblib.dump(model, "../models/model.pkl")
        joblib.dump(scaler, "../models/scaler.pkl")

        # Log model
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train()