"""
predict.py — Modul Inferensi untuk Deteksi Phishing URL
UAS Pembelajaran Mesin | A11.2024.15574 — Mayolus Gavin

Memuat model terbaik dan melakukan prediksi pada data baru.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Tambahkan parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(model_dir='models'):
    """
    Memuat model terbaik, scaler, dan fitur terpilih.
    
    Returns:
        model, scaler, selected_features, model_info
    """
    model = joblib.load(os.path.join(model_dir, 'best_phishing_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    selected_features = joblib.load(os.path.join(model_dir, 'selected_features.joblib'))
    
    with open(os.path.join(model_dir, 'best_model_info.json'), 'r') as f:
        model_info = json.load(f)
    
    return model, scaler, selected_features, model_info


def predict_single(input_data, model, scaler, feature_names, selected_features):
    """
    Prediksi satu sampel data.
    
    Args:
        input_data: dict atau pd.Series dengan fitur URL
        model: Model terlatih
        scaler: StandardScaler
        feature_names: Nama fitur asli (sebelum seleksi)
        selected_features: Nama fitur terpilih
    
    Returns:
        dict: prediction (0/1), label (str), probability (dict)
    """
    # Konversi ke DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Pastikan kolom sesuai
    # Jika belum di-scale, lakukan scaling
    if feature_names:
        # Ambil hanya kolom yang ada di feature_names
        available_cols = [c for c in feature_names if c in df.columns]
        df_features = df[available_cols]
        
        # Scaling
        df_scaled = pd.DataFrame(
            scaler.transform(df_features), 
            columns=available_cols
        )
    else:
        df_scaled = df
    
    # Feature selection
    if selected_features:
        df_selected = df_scaled[[c for c in selected_features if c in df_scaled.columns]]
    else:
        df_selected = df_scaled
    
    # Prediksi
    pred = model.predict(df_selected)
    label = "Phishing" if pred[0] == 1 else "Legitimate"
    
    result = {
        'prediction': int(pred[0]),
        'label': label
    }
    
    # Probabilitas (jika tersedia)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df_selected)
        result['probability'] = {
            'Legitimate': round(float(proba[0][0]), 4),
            'Phishing': round(float(proba[0][1]), 4)
        }
    elif hasattr(model, 'decision_function'):
        decision = model.decision_function(df_selected)
        result['decision_score'] = round(float(decision[0]), 4)
    
    return result


def predict_batch(data_path, model, scaler, feature_names, selected_features):
    """
    Prediksi batch dari file CSV.
    """
    df = pd.read_csv(data_path)
    
    # Drop kolom non-numerik
    cols_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Scaling
    df_scaled = pd.DataFrame(scaler.transform(df_clean), columns=df_clean.columns)
    
    # Feature selection
    df_selected = df_scaled[[c for c in selected_features if c in df_scaled.columns]]
    
    # Prediksi
    preds = model.predict(df_selected)
    labels = ["Phishing" if p == 1 else "Legitimate" for p in preds]
    
    results = pd.DataFrame({
        'prediction': preds,
        'label': labels
    })
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df_selected)
        results['prob_legitimate'] = proba[:, 0].round(4)
        results['prob_phishing'] = proba[:, 1].round(4)
    
    return results


if __name__ == "__main__":
    print("Memuat model...")
    model, scaler, selected_features, model_info = load_model()
    
    print(f"Model: {model_info['model_name']}")
    print(f"Accuracy: {model_info['accuracy']}")
    print(f"Macro-F1: {model_info['macro_f1']}")
    
    # Demo: prediksi dari data test
    data_path = os.path.join('data', 'compressed_PhiUSIIL_Phishing_URL_Dataset.csv')
    df = pd.read_csv(data_path)
    
    # Ambil 5 sampel acak
    samples = df.sample(5, random_state=123)
    feature_names = joblib.load(os.path.join('models', 'feature_names.joblib'))
    
    cols_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    
    print("\nDemo Prediksi (5 sampel):")
    print("-" * 50)
    for i, (idx, row) in enumerate(samples.iterrows()):
        actual = "Phishing" if row['label'] == 1 else "Legitimate"
        input_data = row.drop([c for c in cols_to_drop if c in row.index])
        result = predict_single(input_data, model, scaler, feature_names, selected_features)
        
        status = "✓" if result['label'] == actual else "✗"
        print(f"  [{status}] Sample {i+1}: Actual={actual}, Predicted={result['label']}", end="")
        if 'probability' in result:
            print(f" (Prob: {result['probability']['Phishing']:.2%})")
        else:
            print()
