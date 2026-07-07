"""
ml_core.py — Modul Inti Machine Learning untuk Deteksi Phishing URL
UAS Pembelajaran Mesin | A11.2024.15574 — Mayolus Gavin

Berisi fungsi-fungsi:
- audit_data: Audit dataset lengkap
- preprocess_pipeline: Pipeline preprocessing end-to-end
- train_baseline_models: Training KNN, Naive Bayes, SVM (baseline)
- optimize_models: Optimasi hyperparameter dengan GridSearchCV
- evaluate_model: Evaluasi metrik lengkap
- error_analysis: Analisis kesalahan prediksi
- feature_selection_mi: Seleksi fitur berbasis Mutual Information
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.dummy import DummyClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


# =============================================================================
# 1. AUDIT DATA (Soal 02)
# =============================================================================
def audit_data(df):
    """
    Melakukan audit lengkap terhadap dataset.
    
    Returns:
        dict: Hasil audit (shape, types, missing, duplicates, outlier, 
              target distribution, class imbalance, leakage check)
    """
    audit = {}
    
    # Informasi dasar
    audit['shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}
    audit['column_names'] = list(df.columns)
    
    # Tipe data
    type_counts = df.dtypes.value_counts()
    audit['dtypes'] = {str(k): int(v) for k, v in type_counts.items()}
    audit['dtype_per_column'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values
    missing = df.isnull().sum()
    audit['missing_values'] = {
        'total': int(missing.sum()),
        'per_column': {col: int(v) for col, v in missing.items() if v > 0}
    }
    
    # Duplikat
    audit['duplicates'] = int(df.duplicated().sum())
    
    # Target distribution
    if 'label' in df.columns:
        target_dist = df['label'].value_counts()
        audit['target_distribution'] = {
            str(k): int(v) for k, v in target_dist.items()
        }
        majority = target_dist.max()
        minority = target_dist.min()
        audit['class_imbalance_ratio'] = round(majority / minority, 4)
    
    # Statistik numerik untuk deteksi outlier
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_outliers > 0:
            outlier_info[col] = {
                'count': n_outliers,
                'percentage': round(n_outliers / len(df) * 100, 2),
                'Q1': round(float(Q1), 4),
                'Q3': round(float(Q3), 4),
                'IQR': round(float(IQR), 4)
            }
    audit['outliers'] = outlier_info
    
    # Potensi data leakage
    audit['leakage_check'] = {
        'note': 'Kolom FILENAME, URL, Domain, TLD, Title dihapus karena berpotensi leakage dan tidak relevan sebagai fitur prediktif numerik.',
        'columns_to_drop': ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
    }
    
    return audit


# =============================================================================
# 2. PREPROCESSING PIPELINE (Soal 02)
# =============================================================================
def preprocess_pipeline(df, sample_size=15000, test_size=0.2, random_state=42):
    """
    Pipeline preprocessing end-to-end.
    
    Langkah:
    1. Sampling (untuk efisiensi SVM)
    2. Drop kolom non-numerik
    3. Drop duplikat
    4. Outlier handling (IQR clipping)
    5. StandardScaler
    6. Stratified train-test split
    
    Args:
        df: DataFrame asli
        sample_size: Jumlah baris yang disampling (default 15000)
        test_size: Proporsi data test (default 0.2)
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names, df_before, df_after
    """
    # Simpan data sebelum preprocessing
    df_before = df.describe()
    
    # Langkah 1: Sampling
    if sample_size and len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"[Preprocessing] Sampling: {len(df)} → {len(df_sampled)} baris")
    else:
        df_sampled = df.copy().reset_index(drop=True)
        print(f"[Preprocessing] Menggunakan seluruh data: {len(df_sampled)} baris")
    
    # Langkah 2: Drop kolom non-numerik
    cols_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
    df_cleaned = df_sampled.drop(columns=[c for c in cols_to_drop if c in df_sampled.columns])
    print(f"[Preprocessing] Drop kolom non-numerik: {[c for c in cols_to_drop if c in df_sampled.columns]}")
    
    # Langkah 3: Drop duplikat
    n_before = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df_cleaned)
    print(f"[Preprocessing] Drop duplikat: {n_dropped} baris dihapus, tersisa {len(df_cleaned)} baris")
    
    # Langkah 4: Pisahkan fitur dan target (Hanya gunakan fitur berbasis URL untuk deteksi instan)
    url_features = [
        'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex', 
        'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 
        'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 
        'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL', 
        'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 
        'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
    ]
    X = df_cleaned[[c for c in url_features if c in df_cleaned.columns]]
    y = df_cleaned['label']
    feature_names = list(X.columns)
    
    # Langkah 5: Outlier handling (IQR clipping)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:  # Hanya clip jika ada variasi
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower=lower, upper=upper)
    print(f"[Preprocessing] Outlier clipping (IQR method) selesai")
    
    # Langkah 6: Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    print(f"[Preprocessing] StandardScaler diterapkan pada {len(feature_names)} fitur")
    
    # Simpan data setelah preprocessing
    df_after = X_scaled.describe()
    
    # Langkah 7: Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Preprocessing] Split: Train={len(X_train)}, Test={len(X_test)} (stratified, {int(test_size*100)}%)")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, df_before, df_after


# =============================================================================
# 3. FEATURE SELECTION (Soal 04)
# =============================================================================
def feature_selection_mi(X_train, y_train, X_test, top_k=30):
    """
    Seleksi fitur menggunakan Mutual Information.
    
    Returns:
        X_train_selected, X_test_selected, selected_features, mi_scores
    """
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X_train.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Pilih top-K fitur
    top_k = min(top_k, len(mi_df))
    selected_features = mi_df.head(top_k)['feature'].tolist()
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"[Feature Selection] {len(selected_features)} fitur terpilih dari {len(X_train.columns)} (Mutual Information)")
    
    return X_train_selected, X_test_selected, selected_features, mi_df


# =============================================================================
# 4. EVALUASI MODEL (Soal 03-04)
# =============================================================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluasi model dengan metrik lengkap.
    
    Returns:
        dict: Semua metrik evaluasi
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, average='binary'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='binary'), 4),
        'F1-Score': round(f1_score(y_test, y_pred, average='binary'), 4),
        'Macro-F1': round(f1_score(y_test, y_pred, average='macro'), 4),
        'Balanced Accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # ROC-AUC (jika model mendukung predict_proba)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None
            
        if y_proba is not None:
            metrics['ROC-AUC'] = round(roc_auc_score(y_test, y_proba), 4)
    except Exception:
        metrics['ROC-AUC'] = None
    
    return metrics


# =============================================================================
# 5. BASELINE MODELS (Soal 03)
# =============================================================================
def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Melatih 3 model baseline: KNN, Naive Bayes, SVM.
    
    Returns:
        dict: {model_name: {'model': model_obj, 'metrics': metrics_dict, 'train_time': float}}
    """
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    }
    
    parameters_info = {
        "KNN": "n_neighbors=5, weights='uniform', metric='minkowski' (default). Dipilih karena k=5 adalah standar awal yang stabil.",
        "Naive Bayes": "GaussianNB(var_smoothing=1e-9, default). Asumsi distribusi Gaussian pada fitur.",
        "SVM": "kernel='rbf', C=1.0, gamma='scale'. RBF kernel dipilih karena mampu menangani non-linearity."
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n[Baseline] Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        metrics = evaluate_model(model, X_test, y_test, model_name=f"{name} (Baseline)")
        metrics['Train Time (s)'] = round(train_time, 4)
        metrics['Parameters'] = parameters_info[name]
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'train_time': train_time
        }
        
        print(f"  Accuracy: {metrics['Accuracy']:.4f} | Macro-F1: {metrics['Macro-F1']:.4f} | "
              f"Balanced Acc: {metrics['Balanced Accuracy']:.4f} | Time: {train_time:.2f}s")
    
    return results


# =============================================================================
# 6. OPTIMASI MODEL (Soal 04)
# =============================================================================
def optimize_models(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Optimasi 3 model menggunakan:
    1. GridSearchCV dengan Stratified 5-Fold CV
    2. SMOTE untuk class imbalance handling
    3. Scoring: macro-F1
    
    Returns:
        dict: {model_name: {'model': best_model, 'metrics': metrics, 'best_params': params, 'cv_results': results}}
    """
    
    # Strategi 1: SMOTE untuk class imbalance
    if use_smote and HAS_IMBLEARN:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[Optimasi] SMOTE diterapkan: {len(X_train)} → {len(X_train_res)} baris")
        print(f"  Distribusi sebelum: {dict(pd.Series(y_train).value_counts())}")
        print(f"  Distribusi sesudah: {dict(pd.Series(y_train_res).value_counts())}")
    else:
        X_train_res, y_train_res = X_train, y_train
        if use_smote and not HAS_IMBLEARN:
            print("[Warning] imbalanced-learn tidak terinstall, skip SMOTE")
    
    # Stratified 5-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Parameter grid untuk setiap model
    param_grids = {
        "KNN": {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        "Naive Bayes": {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    }
    
    model_instances = {
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in model_instances.items():
        print(f"\n[Optimasi] GridSearchCV untuk {name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0,
            return_train_score=True
        )
        
        start_time = time.time()
        grid_search.fit(X_train_res, y_train_res)
        train_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test, model_name=f"{name} (Optimized)")
        metrics['Train Time (s)'] = round(train_time, 4)
        metrics['Best Parameters'] = grid_search.best_params_
        metrics['Best CV Score (Macro-F1)'] = round(grid_search.best_score_, 4)
        
        # Cross-validation scores pada data test
        cv_scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring='f1_macro')
        metrics['CV Mean F1'] = round(cv_scores.mean(), 4)
        metrics['CV Std F1'] = round(cv_scores.std(), 4)
        
        results[name] = {
            'model': best_model,
            'metrics': metrics,
            'best_params': grid_search.best_params_,
            'train_time': train_time,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        print(f"  Best Params: {grid_search.best_params_}")
        print(f"  Best CV F1: {grid_search.best_score_:.4f}")
        print(f"  Test Accuracy: {metrics['Accuracy']:.4f} | Macro-F1: {metrics['Macro-F1']:.4f} | "
              f"Balanced Acc: {metrics['Balanced Accuracy']:.4f} | Time: {train_time:.2f}s")
    
    return results


# =============================================================================
# 7. ERROR ANALYSIS (Soal 04)
# =============================================================================
def error_analysis(model, X_test, y_test, feature_names=None, n_samples=10):
    """
    Analisis kesalahan prediksi model.
    
    Returns:
        dict: Informasi error (jumlah, contoh, pola)
    """
    y_pred = model.predict(X_test)
    
    # Indeks kesalahan
    error_mask = y_test.values != y_pred
    n_errors = error_mask.sum()
    n_total = len(y_test)
    
    analysis = {
        'total_errors': int(n_errors),
        'total_samples': int(n_total),
        'error_rate': round(n_errors / n_total * 100, 2),
    }
    
    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    analysis['false_positives'] = int(cm[0, 1])  # Legitimate → Phishing (salah)
    analysis['false_negatives'] = int(cm[1, 0])  # Phishing → Legitimate (salah)
    analysis['true_positives'] = int(cm[1, 1])
    analysis['true_negatives'] = int(cm[0, 0])
    
    # Contoh prediksi salah
    X_errors = X_test[error_mask]
    y_true_errors = y_test[error_mask]
    y_pred_errors = y_pred[error_mask]
    
    if len(X_errors) > 0:
        sample_errors = []
        for i in range(min(n_samples, len(X_errors))):
            idx = X_errors.index[i]
            sample_errors.append({
                'index': int(idx),
                'actual': int(y_true_errors.iloc[i]),
                'predicted': int(y_pred_errors[i]),
                'type': 'False Positive' if y_true_errors.iloc[i] == 0 else 'False Negative'
            })
        analysis['sample_errors'] = sample_errors
    
    # Pola: fitur mana yang paling berbeda pada data salah prediksi
    if feature_names and len(X_errors) > 0:
        X_correct = X_test[~error_mask]
        mean_diff = (X_errors.mean() - X_correct.mean()).abs().sort_values(ascending=False)
        analysis['top_differentiating_features'] = {
            feat: round(float(val), 4) for feat, val in mean_diff.head(10).items()
        }
    
    return analysis


# =============================================================================
# 8. PERBANDINGAN BASELINE vs OPTIMIZED (Soal 04)
# =============================================================================
def compare_baseline_optimized(baseline_results, optimized_results):
    """
    Membuat tabel perbandingan baseline vs optimized.
    
    Returns:
        pd.DataFrame: Tabel komparatif
    """
    rows = []
    for model_name in ['KNN', 'Naive Bayes', 'SVM']:
        # Baseline
        b = baseline_results[model_name]['metrics']
        rows.append({
            'Model': f"{model_name}",
            'Type': 'Baseline',
            'Accuracy': b['Accuracy'],
            'Precision': b['Precision'],
            'Recall': b['Recall'],
            'F1-Score': b['F1-Score'],
            'Macro-F1': b['Macro-F1'],
            'Balanced Acc': b['Balanced Accuracy'],
            'ROC-AUC': b.get('ROC-AUC', '-'),
            'Train Time (s)': b.get('Train Time (s)', '-')
        })
        
        # Optimized
        o = optimized_results[model_name]['metrics']
        rows.append({
            'Model': f"{model_name}",
            'Type': 'Optimized',
            'Accuracy': o['Accuracy'],
            'Precision': o['Precision'],
            'Recall': o['Recall'],
            'F1-Score': o['F1-Score'],
            'Macro-F1': o['Macro-F1'],
            'Balanced Acc': o['Balanced Accuracy'],
            'ROC-AUC': o.get('ROC-AUC', '-'),
            'Train Time (s)': o.get('Train Time (s)', '-')
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# 9. PILIH MODEL TERBAIK (Soal 04-05)
# =============================================================================
def select_best_model(optimized_results):
    """
    Memilih model terbaik berdasarkan kombinasi metrik.
    Kriteria utama: Macro-F1 → Balanced Accuracy → Interpretabilitas
    
    Returns:
        str: Nama model terbaik
        object: Model terbaik
        dict: Metrik model terbaik
    """
    best_name = None
    best_score = -1
    
    for name, result in optimized_results.items():
        m = result['metrics']
        # Skor komposit: 50% Macro-F1 + 30% Balanced Acc + 20% ROC-AUC
        roc = m.get('ROC-AUC', m['Macro-F1']) or m['Macro-F1']
        score = 0.5 * m['Macro-F1'] + 0.3 * m['Balanced Accuracy'] + 0.2 * roc
        
        if score > best_score:
            best_score = score
            best_name = name
    
    best_model = optimized_results[best_name]['model']
    best_metrics = optimized_results[best_name]['metrics']
    
    return best_name, best_model, best_metrics


# =============================================================================
# 10. DUMMY BASELINE (Soal 03)
# =============================================================================
def get_dummy_baseline(X_train, y_train, X_test, y_test):
    """
    Baseline paling sederhana (most_frequent) sebagai pembanding minimal.
    """
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    acc = dummy.score(X_test, y_test)
    return acc
