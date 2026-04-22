import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


# ── GET DATA ──────────────────────────────────────────────────────────────────
def get_data(path):
    df = pd.read_csv(path)
    return df


# ── SPLIT DATA ────────────────────────────────────────────────────────────────
def split_data(df):
    # Replace 'label' with your TARGET column name
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    return X_train, X_test, y_train, y_test


# ── TRAIN MODEL ───────────────────────────────────────────────────────────────
def train_model(X_train, y_train, reg_rate):
    model = LogisticRegression(C=1 / reg_rate, solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ── EVAL MODEL ────────────────────────────────────────────────────────────────
def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_scores)
    return acc, auc


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(args):
    mlflow.sklearn.autolog()

    df = get_data(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train, args.reg_rate)
    acc, auc = eval_model(model, X_test, y_test)

    mlflow.log_metric('val_acc', acc)
    mlflow.log_metric('val_auc', auc)

    print(f'Accuracy : {acc:.4f}')
    print(f'AUC      : {auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, required=True)
    parser.add_argument('--reg_rate',      type=float, default=0.01)
    args = parser.parse_args()

    main(args)
