# UAS Pembelajaran Mesin — Deteksi Phishing URL

## Informasi Mahasiswa
| | |
|---|---|
| **Nama** | Mayolus Gavin |
| **NIM** | A11.2024.15574 |
| **Kelas** | A11.4401 |
| **Dosen** | Junta Zeniarja, M.Kom |
| **Mata Kuliah** | Pembelajaran Mesin |

## Deskripsi Proyek
Proyek ini membangun sistem klasifikasi phishing URL secara end-to-end menggunakan dataset publik PhiUSIIL dari UCI Machine Learning Repository. Sistem ini membandingkan tiga algoritma klasifikasi (KNN, Naive Bayes, SVM) dengan dan tanpa optimasi, serta mengimplementasikan aplikasi web interaktif menggunakan Streamlit.

## Topik
**Optimasi Teknik Machine Learning dalam Klasifikasi Phishing URL**

Sistem ini memprediksi apakah sebuah URL bersifat **Phishing (berbahaya)** atau **Legitimate (aman)** berdasarkan 50+ fitur yang diekstrak dari URL dan konten halaman web.

## Struktur Folder
```
uas ML/
├── data/
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv   # Dataset utama (235K baris)
│   ├── data_dictionary.md                   # Dokumentasi fitur
│   └── source_dataset.md                    # Sumber dan cara akses dataset
├── notebooks/
│   └── uas_ml_phishing_knn_nb_svm_optimization.ipynb  # Notebook eksperimen
├── src/
│   ├── ml_core.py                           # Modul inti ML (preprocessing, training, evaluasi)
│   ├── train.py                             # Pipeline training end-to-end
│   └── predict.py                           # Modul inferensi/prediksi
├── models/
│   ├── best_phishing_model.joblib           # Model terbaik tersimpan
│   ├── scaler.joblib                        # StandardScaler tersimpan
│   ├── feature_names.joblib                 # Daftar fitur
│   ├── selected_features.joblib             # Fitur terpilih
│   └── best_model_info.json                 # Info model terbaik
├── reports/
│   ├── audit_dataset.json                   # Hasil audit data
│   ├── all_experiment_results.csv           # Tabel komparasi eksperimen
│   ├── classification_reports.json          # Laporan klasifikasi
│   └── error_analysis.json                  # Analisis kesalahan
├── app_streamlit.py                         # Aplikasi web Streamlit
├── requirements.txt                         # Dependencies
└── README.md                                # Dokumentasi (file ini)
```

## Cara Menjalankan

### 1. Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Training Pipeline
```bash
cd "uas ML"
python src/train.py
```
Ini akan menjalankan:
- Audit dataset
- Preprocessing (sampling, cleaning, scaling)
- Feature selection (Mutual Information)
- Training baseline (KNN, NB, SVM)
- Optimasi (GridSearchCV + SMOTE + 5-Fold CV)
- Perbandingan baseline vs optimized
- Error analysis
- Simpan model terbaik

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run app_streamlit.py
```

### 4. Jalankan Prediksi Manual
```bash
python src/predict.py
```

## Algoritma yang Digunakan
| Model | Parameter Baseline | Optimasi |
|-------|--------------------|----------|
| **KNN** | n_neighbors=5, uniform, minkowski | GridSearchCV: k=[3,5,7,9,11], weights, metric |
| **Naive Bayes** | GaussianNB(var_smoothing=1e-9) | GridSearchCV: var_smoothing=[1e-11...1e-5] |
| **SVM** | kernel='rbf', C=1.0, gamma='scale' | GridSearchCV: C=[0.1,1,10], gamma, kernel |

## Strategi Optimasi
1. **Hyperparameter Tuning** — GridSearchCV dengan scoring macro-F1
2. **Cross-Validation** — Stratified 5-fold CV
3. **Feature Selection** — Mutual Information top-30 fitur
4. **Class Imbalance Handling** — SMOTE (Synthetic Minority Over-sampling)
5. **Metric Selection** — Macro-F1 sebagai metrik utama

## Metrik Evaluasi
- Accuracy, Precision, Recall, F1-Score
- Macro-F1, Balanced Accuracy
- ROC-AUC
- Confusion Matrix

## Dataset
- **Nama**: PhiUSIIL Phishing URL Dataset
- **Sumber**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- **Ukuran**: 235.795 baris × 56 kolom
- **Random Seed**: 42 (semua proses)

## Referensi
1. Prasad, A., & Chandra, S. (2024). PhiUSIIL Phishing URL Dataset. UCI ML Repository.
2. Setiawan et al. (2026). Imbalanced Multi-class Prediction. ERIES Journal. DOI: 10.7160/eriesj.2026.190106
3. Al Hakim et al. (2026). Optimization of ML Model using Grid and Random Search. JUTIF. DOI: 10.52436/1.jutif.2026.7.3.5627

## Reprodusibilitas
- Semua proses menggunakan `random_state=42`
- Dataset dan model tersimpan di folder `data/` dan `models/`
- Requirements tercatat di `requirements.txt`
