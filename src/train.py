"""
train.py — Pipeline Training End-to-End untuk Deteksi Phishing URL
UAS Pembelajaran Mesin | A11.2024.15574 — Mayolus Gavin

Menjalankan seluruh pipeline:
1. Load & Audit Dataset
2. Preprocessing
3. Feature Selection
4. Baseline Models (KNN, NB, SVM)
5. Optimasi Models (GridSearchCV + SMOTE + CV)
6. Perbandingan Baseline vs Optimized
7. Error Analysis
8. Pilih & Simpan Model Terbaik
9. Export Reports
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import joblib
import pandas as pd
import numpy as np

# Tambahkan parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_core import (
    audit_data, preprocess_pipeline, feature_selection_mi,
    train_baseline_models, optimize_models, evaluate_model,
    error_analysis, compare_baseline_optimized, select_best_model,
    get_dummy_baseline
)


def run_full_pipeline():
    """Menjalankan pipeline training lengkap."""
    
    print("\n" + "=" * 70)
    print("   UAS PEMBELAJARAN MESIN - DETEKSI PHISHING URL")
    print("   A11.2024.15574 - Mayolus Gavin")
    print("=" * 70)
    
    # Setup folder output
    for folder in ['models', 'reports']:
        os.makedirs(folder, exist_ok=True)
    
    # =========================================================================
    # TAHAP 1: LOAD & AUDIT DATASET
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 1] LOAD & AUDIT DATASET")
    print("-" * 50)
    
    data_path = os.path.join('data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    df = pd.read_csv(data_path)
    print(f"Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    
    # Audit
    audit = audit_data(df)
    print(f"Missing values: {audit['missing_values']['total']}")
    print(f"Duplikat: {audit['duplicates']}")
    print(f"Target distribusi: {audit['target_distribution']}")
    print(f"Class imbalance ratio: {audit['class_imbalance_ratio']}")
    print(f"Fitur dengan outlier: {len(audit['outliers'])} kolom")
    
    # Simpan audit
    # Konversi audit dict agar JSON serializable
    audit_serializable = json.loads(json.dumps(audit, default=str))
    with open(os.path.join('reports', 'audit_dataset.json'), 'w') as f:
        json.dump(audit_serializable, f, indent=2, ensure_ascii=False)
    print(">> Audit disimpan di reports/audit_dataset.json")
    
    # =========================================================================
    # TAHAP 2: PREPROCESSING
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 2] PREPROCESSING")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test, scaler, feature_names, df_before, df_after = \
        preprocess_pipeline(df, sample_size=15000, test_size=0.2, random_state=42)
    
    # Simpan scaler
    joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join('models', 'feature_names.joblib'))
    print(">> Scaler disimpan di models/scaler.joblib")
    
    # =========================================================================
    # TAHAP 3: FEATURE SELECTION
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 3] FEATURE SELECTION (Mutual Information)")
    print("-" * 50)
    
    X_train_sel, X_test_sel, selected_features, mi_df = \
        feature_selection_mi(X_train, y_train, X_test, top_k=30)
    
    print("\nTop 10 Fitur (Mutual Information):")
    print(mi_df.head(10).to_string(index=False))
    
    # Simpan fitur terpilih
    joblib.dump(selected_features, os.path.join('models', 'selected_features.joblib'))
    
    # =========================================================================
    # TAHAP 4: DUMMY BASELINE
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 4] DUMMY BASELINE")
    print("-" * 50)
    
    dummy_acc = get_dummy_baseline(X_train_sel, y_train, X_test_sel, y_test)
    print(f"Dummy Baseline Accuracy (most_frequent): {dummy_acc:.4f}")
    print(">> Semua model wajib melampaui skor ini.")
    
    # =========================================================================
    # TAHAP 5: BASELINE MODELS (KNN, NB, SVM)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 5] BASELINE MODELS")
    print("-" * 50)
    
    baseline_results = train_baseline_models(X_train_sel, y_train, X_test_sel, y_test)
    
    # =========================================================================
    # TAHAP 6: OPTIMASI MODELS (GridSearchCV + SMOTE + 5-Fold CV)
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 6] OPTIMASI MODELS (GridSearchCV + SMOTE + 5-Fold CV)")
    print("-" * 50)
    
    optimized_results = optimize_models(X_train_sel, y_train, X_test_sel, y_test, use_smote=True)
    
    # =========================================================================
    # TAHAP 7: PERBANDINGAN BASELINE vs OPTIMIZED
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 7] PERBANDINGAN BASELINE vs OPTIMIZED")
    print("-" * 50)
    
    comparison_df = compare_baseline_optimized(baseline_results, optimized_results)
    print("\n" + comparison_df.to_string(index=False))
    
    # Simpan hasil eksperimen
    comparison_df.to_csv(os.path.join('reports', 'all_experiment_results.csv'), index=False)
    print("\n>> Hasil disimpan di reports/all_experiment_results.csv")
    
    # =========================================================================
    # TAHAP 8: ERROR ANALYSIS
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 8] ERROR ANALYSIS")
    print("-" * 50)
    
    all_error_analysis = {}
    for name in ['KNN', 'Naive Bayes', 'SVM']:
        model = optimized_results[name]['model']
        ea = error_analysis(model, X_test_sel, y_test, feature_names=selected_features)
        all_error_analysis[name] = ea
        print(f"\n{name} (Optimized):")
        print(f"  Total Errors: {ea['total_errors']}/{ea['total_samples']} ({ea['error_rate']}%)")
        print(f"  False Positives: {ea['false_positives']}")
        print(f"  False Negatives: {ea['false_negatives']}")
    
    # =========================================================================
    # TAHAP 9: PILIH & SIMPAN MODEL TERBAIK
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 9] PILIH & SIMPAN MODEL TERBAIK")
    print("-" * 50)
    
    best_name, best_model, best_metrics = select_best_model(optimized_results)
    print(f"\nModel Terbaik: {best_name}")
    print(f"  Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"  Macro-F1: {best_metrics['Macro-F1']:.4f}")
    print(f"  Balanced Accuracy: {best_metrics['Balanced Accuracy']:.4f}")
    print(f"  ROC-AUC: {best_metrics.get('ROC-AUC', 'N/A')}")
    print(f"  Best Params: {best_metrics.get('Best Parameters', 'N/A')}")
    
    # Simpan model terbaik
    model_path = os.path.join('models', 'best_phishing_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"\n>> Model {best_name} disimpan di {model_path}")
    
    # Simpan info model terbaik
    best_info = {
        'model_name': best_name,
        'accuracy': best_metrics['Accuracy'],
        'macro_f1': best_metrics['Macro-F1'],
        'balanced_accuracy': best_metrics['Balanced Accuracy'],
        'roc_auc': best_metrics.get('ROC-AUC'),
        'best_params': best_metrics.get('Best Parameters'),
    }
    with open(os.path.join('models', 'best_model_info.json'), 'w') as f:
        json.dump(best_info, f, indent=2, default=str)
    
    # =========================================================================
    # TAHAP 10: SIMPAN CLASSIFICATION REPORTS
    # =========================================================================
    print("\n" + "-" * 50)
    print("[TAHAP 10] SIMPAN SEMUA REPORTS")
    print("-" * 50)
    
    # Classification reports
    cls_reports = {}
    for name in ['KNN', 'Naive Bayes', 'SVM']:
        cls_reports[f"{name}_baseline"] = baseline_results[name]['metrics']['Classification Report']
        cls_reports[f"{name}_optimized"] = optimized_results[name]['metrics']['Classification Report']
    
    with open(os.path.join('reports', 'classification_reports.json'), 'w') as f:
        json.dump(cls_reports, f, indent=2, default=str)
    
    # Error analysis
    with open(os.path.join('reports', 'error_analysis.json'), 'w') as f:
        json.dump(all_error_analysis, f, indent=2, default=str)
    
    print(">> classification_reports.json disimpan")
    print(">> error_analysis.json disimpan")
    
    # =========================================================================
    # DEMO PREDIKSI
    # =========================================================================
    print("\n" + "-" * 50)
    print("[DEMO] PREDIKSI SAMPLE")
    print("-" * 50)
    
    sample = X_test_sel.iloc[[0]]
    pred = best_model.predict(sample)
    label = "PHISHING" if pred[0] == 1 else "LEGITIMATE"
    print(f"Data index {sample.index[0]} >> Prediksi: {label}")
    
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(sample)
        print(f"Probabilitas: Legitimate={proba[0][0]:.4f}, Phishing={proba[0][1]:.4f}")
    
    print("\n" + "=" * 70)
    print("   PIPELINE SELESAI - Semua output tersimpan di models/ dan reports/")
    print("=" * 70)
    
    return baseline_results, optimized_results, best_name, best_model


if __name__ == "__main__":
    run_full_pipeline()
