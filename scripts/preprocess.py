"""Module untuk Preprocessing Data Bangunan."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import neighbor_feature_names

BASE_NUM_COLS = [
    "count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
    "district_id", "vdcmun_id", "ward_id",
    # fitur turunan per-baris
    "height_per_floor", "area_per_floor", "volume_estimate", "age_per_floor",
    "age_is_outlier", "age_clean", "log_area", "slenderness", "aspect_ratio",
    # statistik geografis (GeoStatsTransformer)
    "ward_id_age_clean_mean", "ward_id_plinth_area_sq_ft_mean",
    "ward_id_height_ft_pre_eq_mean", "ward_id_count_floors_pre_eq_mean",
    "ward_id_share", "rel_height_ward_id", "rel_area_ward_id",
    "vdcmun_id_age_clean_mean", "vdcmun_id_plinth_area_sq_ft_mean",
    "vdcmun_id_height_ft_pre_eq_mean", "vdcmun_id_count_floors_pre_eq_mean",
    "vdcmun_id_share", "rel_height_vdcmun_id", "rel_area_vdcmun_id",
]

CAT_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]


def get_column_lists(use_id_features: bool = True) -> tuple:
    """Mengembalikan daftar kolom numerik dan kategorik yang dipakai model.

    Parameters:
        use_id_features (bool): Jika True, sertakan fitur tetangga building_id.

    Returns:
        tuple: (num_cols, cat_cols).
    """
    num_cols = list(BASE_NUM_COLS)
    if use_id_features:
        num_cols += neighbor_feature_names()
    return num_cols, list(CAT_COLS)


def get_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Membuat dan mengembalikan objek ColumnTransformer untuk preprocessing data.

    Parameters:
        num_cols (list): Daftar nama kolom fitur numerik.
        cat_cols (list): Daftar nama kolom fitur kategorik.

    Returns:
        ColumnTransformer: Preprocessor terintegrasi menggunakan scikit-learn.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )
