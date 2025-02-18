"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
from dotenv import load_dotenv
import argparse
from loguru import logger
import pathlib

import pandas as pd

from joblib import dump
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn

from src.pipeline.build_pipeline import split_train_test, create_pipeline
from src.models.train_evaluate import evaluate_model


# ENVIRONMENT CONFIGURATION ---------------------------

logger.add("recording.log", rotation="500 MB")
load_dotenv()

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
parser.add_argument(
    "--max_features",
    type=str, default="sqrt",
    choices=['sqrt', 'log2'],
    help="Number of features to consider when looking for the best split"
)
parser.add_argument(
    "--experiment_name", type=str, default="titanicml", help="MLFlow experiment name"
)
args = parser.parse_args()

URL_RAW = "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw/data.csv"

n_trees = args.n_trees
jeton_api = os.environ.get("JETON_API", "")
data_path = os.environ.get("data_path", URL_RAW)
data_train_path = os.environ.get("train_path", "data/derived/train.parquet")
data_test_path = os.environ.get("test_path", "data/derived/test.parquet")
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

if jeton_api.startswith("$"):
    logger.info("API token has been configured properly")
else:
    logger.warning("API token has not been configured")


# IMPORT ET STRUCTURATION DONNEES --------------------------------

p = pathlib.Path("data/derived/")
p.mkdir(parents=True, exist_ok=True)

TrainingData = pd.read_csv(data_path)

X_train, X_test, y_train, y_test = split_train_test(
    TrainingData, test_size=0.1,
    train_path=data_train_path,
    test_path=data_test_path
)


# PIPELINE ----------------------------


# Create the pipeline
pipe = create_pipeline(
    n_trees, max_depth=MAX_DEPTH, max_features=MAX_FEATURES
)


mlflow_experiment_name = args.experiment_name
mlflow.set_experiment(
    experiment_name=mlflow_experiment_name
)

# ESTIMATION ET EVALUATION ----------------------

param_grid = {
    "classifier__n_estimators": [10, 20, 50],
    "classifier__max_leaf_nodes": [5, 10, 50],
}

pipe_cross_validation = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring=["accuracy", "precision", "recall", "f1"],
    refit="f1",
    cv=5,
    n_jobs=5,
    verbose=1,
)

pipe_cross_validation.fit(X_train, y_train)
pipe = pipe_cross_validation.best_estimator_

dump(pipe, 'model.joblib')


# Evaluate the model
score, matrix = evaluate_model(pipe, X_test, y_test)

logger.success(f"{score:.1%} de bonnes réponses sur les données de test pour validation")
logger.debug(20 * "-")
logger.info("Matrice de confusion")
logger.debug(matrix)


# LOGGING IN MLFLOW -----------------

input_data_mlflow = mlflow.data.from_pandas(
    TrainingData, source=data_path, name="Raw dataset"
)
training_data_mlflow = mlflow.data.from_pandas(
    pd.concat([X_train, y_train], axis = 1), source=data_path, name="Training data"
)


with mlflow.start_run():

    # Log datasets
    mlflow.log_input(input_data_mlflow, context="raw")
    mlflow.log_input(training_data_mlflow, context="raw")

    # Log parameters
    mlflow.log_param("n_trees", n_trees)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("max_features", MAX_FEATURES)

    # Log best hyperparameters from GridSearchCV
    best_params = pipe_cross_validation.best_params_
    for param, value in best_params.items():
        mlflow.log_param(param, value)

    # Log metrics
    mlflow.log_metric("accuracy", score)

    # Log confusion matrix as an artifact
    matrix_path = "confusion_matrix.txt"
    with open(matrix_path, "w") as f:
        f.write(str(matrix))
    mlflow.log_artifact(matrix_path)

    # Log model
    mlflow.sklearn.log_model(pipe, "model")
