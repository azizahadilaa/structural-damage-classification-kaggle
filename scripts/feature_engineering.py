"""Module untuk Feature Engineering Data Bangunan (Nepal Earthquake Damage).

Berisi tiga komponen yang semuanya kompatibel dengan ``sklearn.pipeline.Pipeline``:

1. ``add_engineered_features``  : fitur turunan per-baris (tanpa label, tanpa state).
2. ``GeoStatsTransformer``      : statistik level ward/vdcmun (tanpa label, di-fit di train).
3. ``NeighborTargetFeatures``   : fitur tetangga berdasarkan urutan ``building_id``
   (memakai label train, dihitung secara leave-one-out agar tidak bocor).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------------------------------------------------------- #
# 1. Fitur turunan per-baris
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur turunan fisik bangunan (rasio, volume, kelangsingan, dst).

    Parameters:
        df (pd.DataFrame): DataFrame input (train/test).

    Returns:
        pd.DataFrame: Salinan DataFrame dengan kolom fitur baru.
    """
    df = df.copy()

    # Mencegah pembagian dengan angka 0
    floors = df["count_floors_pre_eq"].replace(0, 1)

    df["height_per_floor"] = df["height_ft_pre_eq"] / floors
    df["area_per_floor"] = df["plinth_area_sq_ft"] / floors
    df["volume_estimate"] = df["plinth_area_sq_ft"] * df["height_ft_pre_eq"]
    df["age_per_floor"] = df["age_building"] / floors

    # age_building memiliki nilai sentinel (mis. 999) -> anggap outlier/missing
    df["age_is_outlier"] = (df["age_building"] >= 200).astype(int)
    df["age_clean"] = df["age_building"].where(df["age_building"] < 200)

    # Bentuk bangunan: bangunan tinggi-ramping lebih rentan terhadap gempa
    df["log_area"] = np.log1p(df["plinth_area_sq_ft"])
    df["slenderness"] = df["height_ft_pre_eq"] / np.sqrt(df["plinth_area_sq_ft"] + 1)
    df["aspect_ratio"] = df["plinth_area_sq_ft"] / (df["height_ft_pre_eq"] ** 2 + 1)

    return df


# --------------------------------------------------------------------------- #
# 2. Statistik geografis (tanpa label)
# --------------------------------------------------------------------------- #
class GeoStatsTransformer(BaseEstimator, TransformerMixin):
    """Menghitung statistik rata-rata fisik bangunan per ward dan per vdcmun.

    Statistik di-fit hanya pada data train lalu dipakai ulang pada data lain,
    sehingga skala fitur konsisten antara train dan test. Tidak memakai label.
    """

    GROUPS = ("ward_id", "vdcmun_id")
    VALUE_COLS = (
        "age_clean",
        "plinth_area_sq_ft",
        "height_ft_pre_eq",
        "count_floors_pre_eq",
    )

    def fit(self, X: pd.DataFrame, y=None):
        """Mempelajari rata-rata & proporsi bangunan tiap grup dari data train.

        Parameters:
            X (pd.DataFrame): Data train (sudah melalui add_engineered_features).
            y: Diabaikan.

        Returns:
            GeoStatsTransformer: objek ini (sudah di-fit).
        """
        self.maps_ = {}
        self.global_ = {c: X[c].mean() for c in self.VALUE_COLS}
        for g in self.GROUPS:
            stats = X.groupby(g)[list(self.VALUE_COLS)].mean()
            stats["share"] = X.groupby(g).size() / len(X)
            self.maps_[g] = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Menambahkan statistik grup dan selisih relatif terhadap rata-rata grup.

        Parameters:
            X (pd.DataFrame): Data yang akan ditransformasi.

        Returns:
            pd.DataFrame: DataFrame dengan kolom statistik geografis tambahan.
        """
        X = X.copy()
        for g, stats in self.maps_.items():
            mapped = stats.reindex(X[g].values)
            for c in self.VALUE_COLS:
                col = mapped[c].to_numpy()
                X[f"{g}_{c}_mean"] = np.where(np.isnan(col), self.global_[c], col)
            X[f"{g}_share"] = mapped["share"].fillna(0).to_numpy()
            X[f"rel_height_{g}"] = X["height_ft_pre_eq"] - X[f"{g}_height_ft_pre_eq_mean"]
            X[f"rel_area_{g}"] = X["plinth_area_sq_ft"] - X[f"{g}_plinth_area_sq_ft_mean"]
        return X


# --------------------------------------------------------------------------- #
# 3. Fitur tetangga berdasarkan building_id (memakai label train)
# --------------------------------------------------------------------------- #
class NeighborTargetFeatures(BaseEstimator, TransformerMixin):
    """Fitur dari damage_grade bangunan yang ID-nya berdekatan.

    Bangunan dengan ``building_id`` berurutan cenderung disurvei bersama sehingga
    tingkat kerusakannya mirip (di data ini ~52% tetangga langsung punya grade
    yang sama, vs ~24% jika acak). Untuk tiap bangunan dihitung rata-rata grade
    dan proporsi kelas dari ``k`` tetangga sebelum/sesudahnya di data train.

    Saat ``fit_transform`` (data train) bangunan itu sendiri dikeluarkan
    (leave-one-out), sehingga distribusinya mirip dengan data test dan tidak
    terjadi kebocoran label.
    """

    def __init__(self, windows=(1, 3, 10, 30)):
        """Inisialisasi transformer.

        Parameters:
            windows (tuple): Ukuran jendela tetangga (jumlah bangunan per sisi).
        """
        self.windows = windows

    @staticmethod
    def _ids(X: pd.DataFrame) -> np.ndarray:
        """Mengubah kolom building_id menjadi array integer 64-bit."""
        return X["building_id"].astype(np.int64).to_numpy()

    def fit(self, X: pd.DataFrame, y):
        """Menyimpan ID dan label train yang sudah diurutkan.

        Parameters:
            X (pd.DataFrame): Data train yang memuat kolom ``building_id``.
            y: Label kelas 0..4 (array/Series).

        Returns:
            NeighborTargetFeatures: objek ini (sudah di-fit).
        """
        ids = self._ids(X)
        order = np.argsort(ids)
        self.ref_ids_ = ids[order]
        self.ref_y_ = np.asarray(y)[order].astype(int)
        self.cum_ = np.vstack(
            [np.zeros((1, 5)), np.cumsum(np.eye(5)[self.ref_y_], axis=0)]
        )
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit lalu transformasi data train dengan skema leave-one-out.

        Parameters:
            X (pd.DataFrame): Data train.
            y: Label kelas 0..4.

        Returns:
            pd.DataFrame: Data train + fitur tetangga (tanpa membocorkan label sendiri).
        """
        return self.fit(X, y)._build(X, self_in_ref=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformasi data baru (test/validasi) memakai label train.

        Parameters:
            X (pd.DataFrame): Data yang memuat kolom ``building_id``.

        Returns:
            pd.DataFrame: Data + fitur tetangga.
        """
        return self._build(X, self_in_ref=False)

    def _build(self, X: pd.DataFrame, self_in_ref: bool) -> pd.DataFrame:
        """Menghitung semua fitur tetangga untuk X.

        Parameters:
            X (pd.DataFrame): Data input.
            self_in_ref (bool): True jika X adalah data yang sama dengan referensi.

        Returns:
            pd.DataFrame: Salinan X dengan kolom fitur tetangga.
        """
        ids = self._ids(X)
        ref, n = self.ref_ids_, len(self.ref_ids_)
        pos = np.searchsorted(ref, ids)  # posisi sisip (kiri)
        nxt = pos + 1 if self_in_ref else pos
        prv = pos - 1
        grades = np.arange(1, 6)
        out = {}

        for k in self.windows:
            lo_p, hi_p = np.clip(prv - k + 1, 0, n), np.clip(prv + 1, 0, n)
            lo_n, hi_n = np.clip(nxt, 0, n), np.clip(nxt + k, 0, n)
            cp = self.cum_[hi_p] - self.cum_[lo_p]
            cn = self.cum_[hi_n] - self.cum_[lo_n]
            cnt_p, cnt_n = hi_p - lo_p, hi_n - lo_n
            for name, c, cnt in (("prev", cp, cnt_p), ("next", cn, cnt_n)):
                mean = (c * grades).sum(1) / np.maximum(cnt, 1)
                out[f"nb_{name}{k}_mean"] = np.where(cnt > 0, mean, np.nan)
            if k >= 10:
                tot, cnt = cp + cn, np.maximum(cnt_p + cnt_n, 1)
                for j in range(5):
                    out[f"nb_g{j + 1}_w{k}"] = tot[:, j] / cnt

        ip, inx = np.clip(prv, 0, n - 1), np.clip(nxt, 0, n - 1)
        out["nb_prev1_grade"] = self.ref_y_[ip] + 1
        out["nb_next1_grade"] = self.ref_y_[inx] + 1
        out["nb_gap_prev"] = np.log1p(np.abs(ids - ref[ip]))
        out["nb_gap_next"] = np.log1p(np.abs(ref[inx] - ids))

        # Komponen numerik dari ID itu sendiri (prefix/serial survei)
        s = X["building_id"].astype(str).str.zfill(12)
        out["id_prefix4"] = s.str[:4].astype(int).to_numpy()
        out["id_prefix6"] = s.str[:6].astype(int).to_numpy()
        out["id_serial"] = s.str[6:].astype(int).to_numpy()

        res = X.copy()
        for name, val in out.items():
            res[name] = val
        return res


def neighbor_feature_names(windows=(1, 3, 10, 30)) -> list:
    """Mengembalikan daftar nama kolom yang dihasilkan NeighborTargetFeatures.

    Parameters:
        windows (tuple): Ukuran jendela yang sama dengan yang dipakai transformer.

    Returns:
        list: Nama-nama kolom fitur tetangga (urutan tidak berpengaruh).
    """
    names = []
    for k in windows:
        names += [f"nb_prev{k}_mean", f"nb_next{k}_mean"]
        if k >= 10:
            names += [f"nb_g{j}_w{k}" for j in range(1, 6)]
    names += ["nb_prev1_grade", "nb_next1_grade", "nb_gap_prev", "nb_gap_next"]
    names += ["id_prefix4", "id_prefix6", "id_serial"]
    return names
