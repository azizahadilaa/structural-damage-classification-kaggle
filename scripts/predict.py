"""Module untuk Melakukan Prediksi Data Test dan Menghasilkan Submission."""

import os

import joblib
import numpy as np
import pandas as pd


def generate_submission(
    test_path: str, model_path: str, submission_output_path: str
) -> None:
    """Memuat model, memprediksi data test, dan menyimpan submission.csv.

    Kolom ``damage_grade`` ditulis sebagai angka murni (1, 2, 3, 4, 5)
    sesuai spesifikasi evaluator Kaggle.

    Parameters:
        test_path (str): Path ke file test.csv.
        model_path (str): Path ke file model .pkl (bundle pipeline + bobot kelas).
        submission_output_path (str): Path file tujuan submission.csv.
    """
    print("Memuat data test dan model...")
    test_df = pd.read_csv(test_path, dtype={"building_id": str})
    bundle = joblib.load(model_path)
    pipeline, weights = bundle["pipeline"], bundle["class_weights"]

    print("Membuat prediksi...")
    # Pipeline sudah memuat feature engineering; building_id tetap dibutuhkan
    proba = pipeline.predict_proba(test_df)
    predictions = (proba * weights).argmax(axis=1) + 1  # 0..4 -> 1..5

    sub = pd.DataFrame(
        {"building_id": test_df["building_id"], "damage_grade": predictions.astype(int)}
    )

    os.makedirs(os.path.dirname(submission_output_path) or ".", exist_ok=True)
    sub.to_csv(submission_output_path, index=False)
    print(f"Submission disimpan di: {submission_output_path}")
    print(sub["damage_grade"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    generate_submission(
        test_path="../data/test.csv",
        model_path="../models/pipeline_model_lgbm.pkl",
        submission_output_path="../submissions/submission.csv",
    )
