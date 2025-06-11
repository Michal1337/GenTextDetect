import csv
import os
from typing import Dict, Union

import numpy as np
import pandas as pd
import pypickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ex_params import CHECKPOINTS_PATH, DATASETS, DATASETS_PATH, TRAINING_HISTORY_PATH


def eval_model(
    model: Union[LogisticRegression, RandomForestClassifier, xgb.XGBClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    # Evaluate the model on the training set
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    return {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "train_auc": roc_auc_score(y_train, y_train_proba),
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "val_balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
        "val_precision": precision_score(y_val, y_val_pred),
        "val_recall": recall_score(y_val, y_val_pred),
        "val_f1": f1_score(y_val, y_val_pred),
        "val_auc": roc_auc_score(y_val, y_val_proba),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "test_auc": roc_auc_score(y_test, y_test_proba),
    }


if __name__ == "__main__":

    history_path = TRAINING_HISTORY_PATH + f"feature_based/logs.csv"
    if not os.path.exists(history_path):
        with open(history_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "dataset",
                    "feature_set",
                    "train_accuracy",
                    "train_balanced_accuracy",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "train_auc",
                    "val_accuracy",
                    "val_balanced_accuracy",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                    "val_auc",
                    "test_accuracy",
                    "test_balanced_accuracy",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_auc",
                ],
            )
            writer.writeheader()

    for name, config in DATASETS.items():
        if name == "master-testset" or "detect" in name:
            continue
        print(f"Training dataset {name}...")
        max_tokens, cols_c0, reverse_labels = (
            config["num_tokens"],
            config["cols_c0"],
            config["reverse_labels"],
        )

        df_train_path = DATASETS_PATH + f"{name}/train_features2.csv"
        df_val_path = DATASETS_PATH + f"{name}/val_features2.csv"
        df_test_path = DATASETS_PATH + "master-testset/test_features2.csv"

        df_train = pd.read_csv(df_train_path)
        df_val = pd.read_csv(df_val_path)
        df_test = pd.read_csv(df_test_path)

        X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
        X_val, y_val = df_val.drop(columns=["label"]), df_val["label"]
        X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]

        X_train.fillna(0, inplace=True)
        X_val.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train_s, y_train)
        lr_results = eval_model(
            lr_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
        )

        record = {
            "model": "lr",
            "dataset": name,
            "feature_set": 2,
            **lr_results,
        }

        with open(history_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        pypickle.save(
            CHECKPOINTS_PATH + f"feature_based/lr_model_{name}_2.pkl", lr_model
        )

        # Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train_s, y_train)
        rf_results = eval_model(
            rf_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
        )
        record = {
            "model": "rf",
            "dataset": name,
            "feature_set": 2,
            **rf_results,
        }
        with open(history_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        pypickle.save(
            CHECKPOINTS_PATH + f"feature_based/rf_model_{name}_2.pkl", rf_model
        )

        # XGBoost
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train_s, y_train)
        xgb_results = eval_model(
            xgb_model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
        )
        record = {
            "model": "xgb",
            "dataset": name,
            "feature_set": 2,
            **xgb_results,
        }
        with open(history_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)

        pypickle.save(
            CHECKPOINTS_PATH + f"feature_based/xgb_model_{name}_2.pkl", xgb_model
        )
