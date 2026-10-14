# Structural Damage Classification (Nepal Earthquake) - Kaggle

Klasifikasi tingkat kerusakan bangunan (`damage_grade` 1-5) pascagempa Nepal dari fitur struktural bangunan.
Model: **LightGBM** (classical ML, tanpa Neural Network) dalam `scikit-learn Pipeline` + `ColumnTransformer`.

| Metrik | Hasil |
|---|---|
| Macro F1 validasi (80/20) | 0.5934 |
| Public Score Kaggle (macro F1) | **0.59886** (naik dari 0.51166 pada baseline awal) |
| Private Score | diisi setelah kompetisi berakhir |

## Struktur Repository
```
structural-damage-classification-kaggle/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                  # dataset Kaggle (tidak di-commit)
├── notebooks/
│   ├── 01_eda.ipynb       # eksplorasi data
│   ├── 02_modeling.ipynb  # validasi, ablation, evaluasi, tuning
│   └── 03_submission.ipynb# membuat & memeriksa submission
├── scripts/
│   ├── preprocess.py          # ColumnTransformer (imputer, scaler, one-hot)
│   ├── feature_engineering.py # fitur turunan + transformer kustom
│   ├── train.py               # training, validasi, simpan model
│   └── predict.py             # prediksi test -> submission.csv
├── models/                # model tersimpan (.pkl)
├── submissions/           # submission.csv
└── reports/figures/       # gambar analisis
```

## Cara Mengunduh Data
1. Daftar/login di Kaggle, buka halaman kompetisi **Structural Damage Classification**, klik **Join Competition** lalu **I Understand and Accept**.
2. Tab **Data** -> **Download All**, atau lewat CLI:
   ```bash
   kaggle competitions download -c structural-damage-classification
   ```
3. Letakkan `train.csv`, `test.csv`, `sample_submission.csv`, `metadata.csv` di folder `data/`.

## Cara Menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

cd scripts
python train.py                 # evaluasi validasi + latih model final -> ../models/pipeline_model_lgbm.pkl
python predict.py               # -> ../submissions/submission.csv
```
Skrip dijalankan dari folder `scripts/` karena memakai path relatif. Notebook dijalankan dari folder `notebooks/` (urutan 01 -> 02 -> 03; notebook 03 membutuhkan model dari `train.py`).
Training memakan waktu beberapa menit hingga belasan menit tergantung CPU.

## Workflow Analisis
1. **EDA** (`01_eda.ipynb`): distribusi target (tidak seimbang, Grade 5 ≈ 36%), missing value, korelasi numerik, dan temuan pola pada `building_id`.
2. **Preprocessing** (`preprocess.py`): `ColumnTransformer` - numerik: `SimpleImputer(median)` + `StandardScaler`; kategorik: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
3. **Feature engineering** (`feature_engineering.py`), semuanya di dalam `Pipeline`:
   - Fitur fisik: tinggi/luas per lantai, volume, kelangsingan, rasio aspek, usia per lantai, `age_clean` (nilai sentinel usia ≥ 200 dianggap outlier).
   - `GeoStatsTransformer`: rata-rata karakteristik bangunan per ward dan per vdcmun (di-fit hanya di train, tanpa label).
   - `NeighborTargetFeatures`: fitur dari `damage_grade` bangunan dengan `building_id` berdekatan (lihat bagian Interpretasi).
4. **Model**: `LGBMClassifier` (700 pohon, lr 0.05, 127 leaves, subsample 0.8, colsample 0.6, lambda 2).
5. **Optimasi keputusan**: bobot probabilitas per kelas yang dicari pada data validasi untuk memaksimalkan **macro F1**.
6. **Evaluasi**: stratified hold-out 80/20; classification report dan confusion matrix di `02_modeling.ipynb`.

## Hasil Validasi

| Setting | Micro F1 | Macro F1 |
|---|---|---|
| Argmax | 0.6243 | 0.5883 |
| + bobot kelas (dipakai untuk submission) | 0.6203 | **0.5934** |

Per kelas (dengan bobot kelas): Grade 1 F1 0.69, Grade 2 0.46, Grade 3 0.49, Grade 4 0.57, Grade 5 0.76.

Ablation fitur tetangga `building_id` (setting ringan: LightGBM 150 pohon, lr 0.1; lihat `02_modeling.ipynb`):

| Varian | Micro F1 | Macro F1 |
|---|---|---|
| Tanpa fitur ID | ≈ 0.579 | ≈ 0.532 |
| Dengan fitur ID | ≈ 0.620 | ≈ 0.584 |

## Interpretasi

**Metrik.** Skor public (0.59886) hampir sama dengan macro F1 validasi (0.5934), sehingga metrik Kaggle adalah macro F1 dan tidak ada tanda overfitting ke data validasi.

**Kelas.** Grade 1 dan Grade 5 (kerusakan paling ringan dan paling berat) paling mudah dikenali. Grade 2 dan 3 paling sulit (F1 ≈ 0.46-0.49) karena kondisi di tengah spektrum kerusakan saling tumpang tindih. Bobot kelas optimal (≈ [0.75, 1.18, 1.18, 1.06, 0.83]) menurunkan bobot Grade 1 dan 5 serta menaikkan bobot kelas tengah agar recall kelas sulit membaik, sehingga macro F1 naik (0.5883 -> 0.5934) walau micro F1 turun sedikit.

**Fitur penting** (gain model LightGBM final, lihat `reports/figures/feature_importance.png`):
- Dari 15 fitur teratas, **13 adalah fitur tetangga `building_id`**. Yang paling berpengaruh adalah proporsi Grade 5 dan Grade 1 pada 10 bangunan di sekitar ID (`nb_g5_w10`, `nb_g1_w10`), diikuti proporsi kelas pada jendela 30 bangunan (`nb_g5_w30`, `nb_g4_w10`, dst.) dan rata-rata grade tetangga (`nb_prev10_mean`, `nb_next10_mean`).
- Fitur fisik yang masuk Top 15 hanya **usia bangunan** (`age_building` urutan 11, `age_clean` urutan 15), masing-masing jauh di bawah fitur tetangga teratas.
- Fitur turunan seperti rasio luas/tinggi per lantai dan statistik ward berkontribusi lebih kecil dan tidak masuk Top 15. Secara fisik masuk akal bahwa bangunan yang lebih tua cenderung lebih rentan.
- Artinya model terutama belajar dari **konteks lokasi survei** (kerusakan bangunan di sekitarnya), dan sifat fisik bangunan hanya berperan sebagai pelengkap.

**Fitur tetangga `building_id` - catatan penting.**
- Pada data train, 51.9% bangunan dengan ID berurutan memiliki grade yang sama (peluang acak 24.5%), karena bangunan dengan ID berdekatan disurvei bersamaan di area yang sama.
- Split train/test pada kompetisi ini acak, sehingga tetangga bangunan test hampir selalu ada di train dan labelnya bisa dipakai sebagai fitur.
- Fitur dihitung di dalam `Pipeline` dengan skema leave-one-out untuk data train (bangunan itu sendiri tidak dihitung), dan data test hanya memakai label train, sehingga tidak ada kebocoran label test.
- **Keterbatasan:** fitur ini memanfaatkan struktur ID survei, bukan sifat fisik bangunan, sehingga kemungkinan besar tidak membantu pada gempa/data baru. Tanpa fitur ini macro F1 validasi ≈ 0.53, sehingga sebagian besar peningkatan dari baseline berasal dari fitur ini. Untuk menonaktifkan, set `USE_ID_FEATURES = False` di `scripts/train.py`.

## Aturan Tugas
- Tanpa Neural Network (LightGBM; XGBoost tersedia untuk ensemble opsional).
- Preprocessing, feature engineering, dan tuning memakai `Pipeline` + `ColumnTransformer`.
- Setiap fungsi/class di `scripts/` punya docstring.
- Dataset tidak di-commit (`data/*.csv` di `.gitignore`).
- `damage_grade` pada `submission.csv` berupa angka 1-5.
