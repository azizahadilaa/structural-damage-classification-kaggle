"""Module untuk Training Model Supervised Machine Learning (LightGBM / Ensemble)."""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from feature_engineering import (
    GeoStatsTransformer,
    NeighborTargetFeatures,
    add_engineered_features,
)
from preprocess import get_column_lists, get_preprocessor

# ----------------------------- KONFIGURASI -------------------------------- #
USE_ID_FEATURES = True   # fitur tetangga building_id (lihat README / catatan di bawah)
MODEL_TYPE = "lgbm"      # "lgbm" (teruji) atau "ensemble" (LGBM + XGB, belum teruji)
METRIC = "macro"         # sesuaikan dengan metrik Kaggle: "macro" atau "micro"
N_JOBS = -1
SEED = 42
# -------------------------------------------------------------------------- #


def load_train(train_path: str) -> tuple:
    """Membaca train.csv dan mengubah target 'Grade k' menjadi integer 0..4.

    Parameters:
        train_path (str): Path file train.csv.

    Returns:
        tuple: (X, y) dengan X DataFrame fitur (termasuk building_id) dan y array 0..4.
    """
    df = pd.read_csv(train_path, dtype={"building_id": str})
    y = df["damage_grade"].str.extract(r"(\d)")[0].astype(int).to_numpy() - 1
    return df.drop(columns=["damage_grade"]), y


def build_model(model_type: str = MODEL_TYPE):
    """Membuat estimator akhir (LightGBM atau ensemble LightGBM + XGBoost).

    Parameters:
        model_type (str): "lgbm" atau "ensemble".

    Returns:
        estimator: Classifier scikit-learn compatible dengan predict_proba.
    """
    lgbm = LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    if model_type == "lgbm":
        return lgbm
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.08,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=N_JOBS,
        eval_metric="mlogloss",
    )
    return VotingClassifier(
        estimators=[("lgbm", lgbm), ("xgb", xgb)], voting="soft", weights=[2, 1]
    )


def build_pipeline(use_id_features: bool = USE_ID_FEATURES, model_type: str = MODEL_TYPE):
    """Merakit Pipeline lengkap: feature engineering -> ColumnTransformer -> model.

    Parameters:
        use_id_features (bool): Aktifkan fitur tetangga building_id.
        model_type (str): "lgbm" atau "ensemble".

    Returns:
        Pipeline: Pipeline scikit-learn yang menerima DataFrame mentah (dengan building_id).
    """
    num_cols, cat_cols = get_column_lists(use_id_features)
    steps = [
        ("fe", FunctionTransformer(add_engineered_features)),
        ("geo", GeoStatsTransformer()),
    ]
    if use_id_features:
        steps.append(("neighbors", NeighborTargetFeatures()))
    steps += [
        ("preprocessor", get_preprocessor(num_cols, cat_cols)),
        ("classifier", build_model(model_type)),
    ]
    return Pipeline(steps=steps)


def score(y_true, y_pred, average: str) -> float:
    """Menghitung F1-score dengan rata-rata tertentu.

    Parameters:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        average (str): "micro" atau "macro".

    Returns:
        float: Nilai F1.
    """
    return f1_score(y_true, y_pred, average=average)


def optimize_class_weights(proba: np.ndarray, y_true: np.ndarray, metric: str = METRIC) -> np.ndarray:
    """Mencari bobot per kelas untuk probabilitas agar F1 validasi maksimum.

    Memakai coordinate ascent sederhana pada pengali probabilitas tiap kelas.

    Parameters:
        proba (np.ndarray): Probabilitas prediksi (n_sampel, 5) pada data validasi.
        y_true (np.ndarray): Label asli 0..4.
        metric (str): "micro" atau "macro".

    Returns:
        np.ndarray: Vektor bobot panjang 5 (dinormalisasi agar rata-rata = 1).
    """
    w = np.ones(proba.shape[1])
    best = score(y_true, (proba * w).argmax(1), metric)
    grid = np.array([0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.4, 1.6])
    for _ in range(3):
        for k in range(len(w)):
            for g in grid:
                w2 = w.copy()
                w2[k] *= g
                s = score(y_true, (proba * w2).argmax(1), metric)
                if s > best + 1e-6:
                    best, w = s, w2
    return w / w.mean()


def train_and_save_model(
    train_path: str, model_output_path: str, figure_output_path: str
) -> None:
    """Evaluasi dengan validation split, lalu melatih model final pada seluruh data.

    Menyimpan bundle (pipeline + bobot kelas) ke .pkl dan grafik feature importance.

    Parameters:
        train_path (str): Path file train.csv.
        model_output_path (str): Path simpan model .pkl.
        figure_output_path (str): Path simpan grafik feature importance.
    """
    print("Memuat data train...")
    X, y = load_train(train_path)

    # 1. EVALUASI: 80% train / 20% validation (pipeline di-fit ulang dari nol)
    print("\n--- Validation split (80/20) ---")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_val)
    pred = proba.argmax(1)
    print(f"Micro F1 (argmax)        : {score(y_val, pred, 'micro'):.4f}")
    print(f"Macro F1 (argmax)        : {score(y_val, pred, 'macro'):.4f}")

    weights = optimize_class_weights(proba, y_val, METRIC)
    pred_w = (proba * weights).argmax(1)
    print(f"Bobot kelas optimal      : {np.round(weights, 3)}")
    print(f"Micro F1 (+bobot kelas)  : {score(y_val, pred_w, 'micro'):.4f}")
    print(f"Macro F1 (+bobot kelas)  : {score(y_val, pred_w, 'macro'):.4f}\n")
    print(classification_report(y_val, pred_w, target_names=[f"Grade {i}" for i in range(1, 6)]))

    # 2. MODEL FINAL: fit ulang pada seluruh data train
    print("Melatih model final pada seluruh data train...")
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump({"pipeline": final_pipe, "class_weights": weights}, model_output_path)
    print(f"Model .pkl disimpan di: {model_output_path}")

    # 3. FEATURE IMPORTANCE
    save_feature_importance(final_pipe, figure_output_path)


def save_feature_importance(pipe: Pipeline, figure_output_path: str) -> None:
    """Menyimpan grafik 15 fitur terpenting dari komponen LightGBM.

    Parameters:
        pipe (Pipeline): Pipeline yang sudah di-fit.
        figure_output_path (str): Path file gambar keluaran.
    """
    os.makedirs(os.path.dirname(figure_output_path) or ".", exist_ok=True)
    try:
        clf = pipe.named_steps["classifier"]
        if hasattr(clf, "named_estimators_"):
            clf = clf.named_estimators_["lgbm"]
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        imp = clf.booster_.feature_importance(importance_type="gain")
        top = np.argsort(imp)[-15:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top)), imp[top], align="center")
        plt.yticks(range(len(top)), [names[i] for i in top])
        plt.title("Top 15 Feature Importance (LightGBM, gain)")
        plt.xlabel("Total gain")
        plt.tight_layout()
        plt.savefig(figure_output_path, dpi=300)
        plt.close()
        print(f"Feature importance disimpan di: {figure_output_path}")
    except Exception as e:  # noqa: BLE001
        print(f"Gagal menyimpan feature importance: {e}")


if __name__ == "__main__":
    train_and_save_model(
        train_path="../data/train.csv",
        model_output_path="../models/pipeline_model_lgbm.pkl",
        figure_output_path="../reports/figures/feature_importance.png",
    )
