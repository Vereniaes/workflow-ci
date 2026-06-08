# MLProject/modelling.py
#
# -> script training untuk MLflow Project (Kriteria 3)
#      -> menerima hyperparameter via argparse (dipakai mlflow run)
#      -> train RandomForestRegressor
#      -> manual logging (bukan autolog):
#           -> params, metrics, model artifact
#      -> output: model tersimpan di mlruns/ atau remote tracking URI
# -> dipanggil oleh: mlflow run MLProject/ -P n_estimators=100

import os
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

# setup logging ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# end of setup logging -----------------------------------------------------------


# konstanta ----------------------------------------------------------------------

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "student_preprocessing", "student_preprocessed.csv"
)
TARGET_COLUMN  = "G3"
EXPERIMENT_NAME = "workflow-ci-student-performance"

# end of konstanta ---------------------------------------------------------------


# helper -------------------------------------------------------------------------

# fungsi untuk load dataset preprocessed
# input  : path - string path ke CSV preprocessed
# output : tuple (X, y) DataFrame dan Series
def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading dataset dari: {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    logger.info(f"Dataset: X={X.shape}, y={y.shape}")
    return X, y


# fungsi untuk hitung dan log semua metrik evaluasi
# input  : y_true, y_pred - array nilai aktual dan prediksi
# output : dict { mae, mse, rmse, r2, mape }
def log_metrics(y_true, y_pred) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE - exclude zero actuals
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mask = y_true_arr != 0
    mape = float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask]))) if mask.sum() > 0 else float("nan")

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        logger.info(f"{k.upper():<6}: {v:.4f}")
    return metrics

# end of helper ------------------------------------------------------------------


# main ---------------------------------------------------------------------------

# fungsi entry point - dipanggil via mlflow run atau langsung
# input  : args dari argparse (n_estimators, max_depth, dll)
# output : None (model + metrics di-log ke MLflow)
def main(args) -> None:
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    X, y = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # jika dipanggil via `mlflow run`, MLFLOW_RUN_ID sudah di-set di env
    # -> skip set_experiment() karena experiment sudah dikelola mlflow run
    # jika dipanggil langsung via python, set experiment manual
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logger.info(f"Run ID: {run.info.run_id}")

        # - log hyperparameter
        mlflow.log_param("model_type",        "RandomForestRegressor")
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         str(args.max_depth))
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
        mlflow.log_param("max_features",      args.max_features)
        mlflow.log_param("test_size",         args.test_size)
        mlflow.log_param("random_state",      args.random_state)

        # train
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # evaluasi dan log metrics
        y_pred = model.predict(X_test)
        metrics = log_metrics(y_test, y_pred)

        # - log model artefak
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        # simpan run_id ke file (untuk step GitHub Actions berikutnya)
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        logger.info("=" * 60)
        logger.info(f"Experiment: {EXPERIMENT_NAME}")
        logger.info(f"Run ID    : {run_id}")
        logger.info(f"R2        : {metrics['r2']:.4f}")
        logger.info(f"RMSE      : {metrics['rmse']:.4f}")
        logger.info("=" * 60)

# end of main --------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest untuk Student Performance")
    parser.add_argument("--n-estimators",      type=int,   default=100,    help="Jumlah trees")
    parser.add_argument("--max-depth",         type=int,   default=10,     help="Max depth (0 = None)")
    parser.add_argument("--min-samples-split", type=int,   default=5,      help="Min samples untuk split")
    parser.add_argument("--min-samples-leaf",  type=int,   default=1,      help="Min samples di leaf")
    parser.add_argument("--max-features",      type=str,   default="sqrt", help="Max features strategy")
    parser.add_argument("--test-size",         type=float, default=0.2,    help="Proporsi test split")
    parser.add_argument("--random-state",      type=int,   default=42,     help="Random state")
    args = parser.parse_args()
    main(args)
