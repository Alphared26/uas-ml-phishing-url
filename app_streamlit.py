"""
app_streamlit.py - Aplikasi Web Deteksi Phishing URL
UAS Pembelajaran Mesin | A11.2024.15574 - Mayolus Gavin

Jalankan dengan: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from urllib.parse import urlparse

# Konfigurasi halaman default
st.set_page_config(
    page_title="Deteksi Phishing URL - UAS ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL & DATA
# =============================================================================
def load_model():
    """Memuat model, scaler, dan fitur."""
    try:
        model = joblib.load('models/best_phishing_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        selected_features = joblib.load('models/selected_features.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        
        with open('models/best_model_info.json', 'r') as f:
            model_info = json.load(f)
        
        return model, scaler, selected_features, feature_names, model_info
    except FileNotFoundError:
        return None, None, None, None, None


def load_reports():
    """Memuat laporan eksperimen."""
    reports = {}
    try:
        with open('reports/audit_dataset.json', 'r') as f:
            reports['audit'] = json.load(f)
    except FileNotFoundError:
        reports['audit'] = None
    
    try:
        reports['experiments'] = pd.read_csv('reports/all_experiment_results.csv')
    except FileNotFoundError:
        reports['experiments'] = None
    
    try:
        with open('reports/classification_reports.json', 'r') as f:
            reports['classification'] = json.load(f)
    except FileNotFoundError:
        reports['classification'] = None
    
    try:
        with open('reports/error_analysis.json', 'r') as f:
            reports['errors'] = json.load(f)
    except FileNotFoundError:
        reports['errors'] = None
    
    return reports


def load_dataset_sample():
    """Memuat sampel dataset untuk visualisasi."""
    try:
        df = pd.read_csv('data/compressed_PhiUSIIL_Phishing_URL_Dataset.csv', nrows=5000)
        return df
    except FileNotFoundError:
        return None


def parse_url_features(url_string):
    """
    Ekstrak fitur-fitur dari URL string mentah secara sederhana.
    """
    features = {}
    
    if not url_string.startswith(('http://', 'https://')):
        url_for_parse = 'http://' + url_string
    else:
        url_for_parse = url_string
        
    try:
        parsed = urlparse(url_for_parse)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
    except Exception:
        domain = ""
        path = ""
        query = ""
        
    features['URLLength'] = len(url_string)
    features['DomainLength'] = len(domain)
    
    # IsDomainIP
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    features['IsDomainIP'] = 1 if re.match(ip_pattern, domain) else 0
    
    # URLSimilarityIndex
    features['URLSimilarityIndex'] = 100.0
    
    # TLD length
    parts = domain.split('.')
    tld = parts[-1] if len(parts) > 1 else ""
    features['TLDLength'] = len(tld)
    
    # No of SubDomain
    features['NoOfSubDomain'] = max(0, len(parts) - 2) if len(parts) > 1 else 0
    
    # Obfuscation
    features['HasObfuscation'] = 1 if '%20' in url_string or '@' in url_string or '\\' in url_string else 0
    features['NoOfObfuscatedChar'] = url_string.count('%') + url_string.count('@')
    features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / len(url_string) if len(url_string) > 0 else 0.0
    
    # Letter & digit counts
    letters = sum(1 for c in url_string if c.isalpha())
    digits = sum(1 for c in url_string if c.isdigit())
    features['NoOfLettersInURL'] = letters
    features['LetterRatioInURL'] = letters / len(url_string) if len(url_string) > 0 else 0.0
    features['NoOfDegitsInURL'] = digits
    features['DegitRatioInURL'] = digits / len(url_string) if len(url_string) > 0 else 0.0
    
    # Special characters
    features['NoOfEqualsInURL'] = url_string.count('=')
    features['NoOfQMarkInURL'] = url_string.count('?')
    features['NoOfAmpersandInURL'] = url_string.count('&')
    
    special_chars = sum(1 for c in url_string if not c.isalnum() and c not in ['/', ':', '.', '-', '_'])
    features['NoOfOtherSpecialCharsInURL'] = special_chars
    features['SpacialCharRatioInURL'] = special_chars / len(url_string) if len(url_string) > 0 else 0.0
    
    features['IsHTTPS'] = 1 if url_string.lower().startswith('https') else 0
    
    return features


def get_default_features():
    df = load_dataset_sample()
    if df is not None:
        cols_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        return df_clean[df['label'] == 0].median().to_dict()
    return {}


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("Phishing URL Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi Halaman",
    ["Dashboard", "Prediksi URL", "Perbandingan Model", "Visualisasi", "Tentang"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**UAS Pembelajaran Mesin**  
NIM: A11.2024.15574  
Nama: Mayolus Gavin  
Dosen: Junta Zeniarja, M.Kom
""")

# Load resources
model, scaler, selected_features, feature_names, model_info = load_model()
reports = load_reports()


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "Dashboard":
    st.title("Dashboard Data Phishing URL")
    st.subheader("Informasi Dataset PhiUSIIL Phishing URL")
    st.markdown("---")
    
    if reports.get('audit'):
        audit = reports['audit']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Baris Data", f"{audit['shape']['rows']:,}")
        with col2:
            st.metric("Total Fitur", audit['shape']['columns'])
        with col3:
            st.metric("Missing Values", audit['missing_values']['total'])
        with col4:
            st.metric("Data Duplikat", audit['duplicates'])
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Distribusi Target (Label)")
            target = audit.get('target_distribution', {})
            fig = px.pie(
                values=list(target.values()),
                names=['Legitimate (0)', 'Phishing (1)'],
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.3
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Audit Kualitas Data")
            st.write(f"Rasio Imbalance Kelas: {audit.get('class_imbalance_ratio', 'N/A')}")
            
            if audit.get('outliers'):
                st.write(f"Jumlah Fitur Mengandung Outlier: {len(audit['outliers'])} kolom")
                outlier_df = pd.DataFrame([
                    {'Fitur': k, 'Jumlah Outlier': v['count'], 'Persentase': f"{v['percentage']}%"}
                    for k, v in list(audit['outliers'].items())[:10]
                ])
                st.dataframe(outlier_df, width='stretch')
    
    # Statistik fitur
    df_sample = load_dataset_sample()
    if df_sample is not None:
        st.markdown("---")
        st.subheader("Distribusi Fitur Numerik Utama")
        
        numeric_cols = ['URLLength', 'DomainLength', 'URLSimilarityIndex', 'CharContinuationRate', 'TLDLength']
        available_cols = [c for c in numeric_cols if c in df_sample.columns]
        
        if available_cols:
            selected_feat = st.selectbox("Pilih Fitur:", available_cols)
            fig = px.histogram(
                df_sample, x=selected_feat, color='label',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'label': 'Label', selected_feat: selected_feat},
                barmode='overlay', opacity=0.7
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')


# =============================================================================
# PAGE 2: PREDIKSI URL
# =============================================================================
elif page == "Prediksi URL":
    st.title("Prediksi Phishing URL")
    st.subheader("Uji URL secara manual atau input fitur")
    st.markdown("---")
    
    if model is None:
        st.error("Model belum dilatih. Harap jalankan file train.py terlebih dahulu.")
    else:
        st.write(f"Model Aktif: {model_info.get('model_name', 'N/A')} | "
                 f"Akurasi: {model_info.get('accuracy', 'N/A')} | "
                 f"Macro-F1: {model_info.get('macro_f1', 'N/A')}")
        
        st.markdown("---")
        
        st.subheader("Uji URL Mentah")
        raw_url = st.text_input("Masukkan alamat URL:", placeholder="Contoh: https://login-paypal-security.com")
        
        default_vals = get_default_features()
        
        if st.button("Ekstrak Fitur URL"):
            if raw_url:
                extracted = parse_url_features(raw_url)
                for k, v in extracted.items():
                    st.session_state[f"feat_{k}"] = v
                st.success("Fitur berhasil diekstrak. Nilai pada form di bawah telah disesuaikan.")
            else:
                st.warning("Silakan masukkan URL.")
        
        st.markdown("---")
        
        with st.form("prediction_form"):
            st.subheader("Detail Fitur Form (Dapat Disesuaikan)")
            
            col1, col2, col3 = st.columns(3)
            
            def get_val(key, default):
                return st.session_state.get(f"feat_{key}", default_vals.get(key, default))
            
            with col1:
                url_length = st.number_input("URL Length", min_value=0, value=int(get_val('URLLength', 50)), step=1)
                domain_length = st.number_input("Domain Length", min_value=0, value=int(get_val('DomainLength', 15)), step=1)
                is_domain_ip = st.selectbox("Is Domain IP?", [0, 1], index=int(get_val('IsDomainIP', 0)))
                url_sim_index = st.slider("URL Similarity Index", 0.0, 100.0, float(get_val('URLSimilarityIndex', 100.0)))
                char_cont_rate = st.slider("Char Continuation Rate", 0.0, 1.0, float(get_val('CharContinuationRate', 0.5)))
                tld_legit_prob = st.slider("TLD Legitimate Prob", 0.0, 1.0, float(get_val('TLDLegitimateProb', 0.5)))
                url_char_prob = st.slider("URL Char Prob", 0.0, 1.0, float(get_val('URLCharProb', 0.05)))
                tld_length = st.number_input("TLD Length", min_value=0, value=int(get_val('TLDLength', 3)), step=1)
                no_subdomain = st.number_input("No of SubDomain", min_value=0, value=int(get_val('NoOfSubDomain', 1)), step=1)
                has_obfuscation = st.selectbox("Has Obfuscation?", [0, 1], index=int(get_val('HasObfuscation', 0)))
            
            with col2:
                no_obfuscated_char = st.number_input("No of Obfuscated Char", min_value=0, value=int(get_val('NoOfObfuscatedChar', 0)), step=1)
                obfuscation_ratio = st.slider("Obfuscation Ratio", 0.0, 1.0, float(get_val('ObfuscationRatio', 0.0)))
                no_letters = st.number_input("No of Letters in URL", min_value=0, value=int(get_val('NoOfLettersInURL', 30)), step=1)
                letter_ratio = st.slider("Letter Ratio in URL", 0.0, 1.0, float(get_val('LetterRatioInURL', 0.7)))
                no_digits = st.number_input("No of Digits in URL", min_value=0, value=int(get_val('NoOfDegitsInURL', 5)), step=1)
                digit_ratio = st.slider("Digit Ratio in URL", 0.0, 1.0, float(get_val('DegitRatioInURL', 0.1)))
                no_equals = st.number_input("No of Equals in URL", min_value=0, value=int(get_val('NoOfEqualsInURL', 0)), step=1)
                no_qmark = st.number_input("No of QMark in URL", min_value=0, value=int(get_val('NoOfQMarkInURL', 0)), step=1)
                no_ampersand = st.number_input("No of Ampersand in URL", min_value=0, value=int(get_val('NoOfAmpersandInURL', 0)), step=1)
                no_special = st.number_input("No of Other Special Chars", min_value=0, value=int(get_val('NoOfOtherSpecialCharsInURL', 3)), step=1)
            
            with col3:
                special_ratio = st.slider("Special Char Ratio", 0.0, 1.0, float(get_val('SpacialCharRatioInURL', 0.1)))
                is_https = st.selectbox("Is HTTPS?", [0, 1], index=int(get_val('IsHTTPS', 1)))
                line_of_code = st.number_input("Line of Code", min_value=0, value=int(get_val('LineOfCode', 100)), step=10)
                largest_line = st.number_input("Largest Line Length", min_value=0, value=int(get_val('LargestLineLength', 500)), step=50)
                has_title = st.selectbox("Has Title?", [0, 1], index=int(get_val('HasTitle', 1)))
                domain_title_match = st.slider("Domain Title Match", 0.0, 1.0, float(get_val('DomainTitleMatchScore', 0.5)))
                url_title_match = st.slider("URL Title Match", 0.0, 1.0, float(get_val('URLTitleMatchScore', 0.3)))
                has_favicon = st.selectbox("Has Favicon?", [0, 1], index=int(get_val('HasFavicon', 1)))
                robots = st.selectbox("Robots?", [0, 1], index=int(get_val('Robots', 1)))
                is_responsive = st.selectbox("Is Responsive?", [0, 1], index=int(get_val('IsResponsive', 1)))
            
            with st.expander("Fitur Tambahan (HTML & Reputasi)"):
                col4, col5 = st.columns(2)
                with col4:
                    no_url_redirect = st.number_input("No of URL Redirect", min_value=0, value=int(get_val('NoOfURLRedirect', 0)), step=1)
                    no_self_redirect = st.number_input("No of Self Redirect", min_value=0, value=int(get_val('NoOfSelfRedirect', 0)), step=1)
                    has_description = st.selectbox("Has Description?", [0, 1], index=int(get_val('HasDescription', 1)))
                    no_popup = st.number_input("No of Popup", min_value=0, value=int(get_val('NoOfPopup', 0)), step=1)
                    no_iframe = st.number_input("No of iFrame", min_value=0, value=int(get_val('NoOfiFrame', 0)), step=1)
                    has_ext_form = st.selectbox("Has External Form Submit?", [0, 1], index=int(get_val('HasExternalFormSubmit', 0)))
                    has_social = st.selectbox("Has Social Net?", [0, 1], index=int(get_val('HasSocialNet', 0)))
                    has_submit = st.selectbox("Has Submit Button?", [0, 1], index=int(get_val('HasSubmitButton', 0)))
                with col5:
                    has_hidden = st.selectbox("Has Hidden Fields?", [0, 1], index=int(get_val('HasHiddenFields', 0)))
                    has_password = st.selectbox("Has Password Field?", [0, 1], index=int(get_val('HasPasswordField', 0)))
                    bank = st.selectbox("Bank?", [0, 1], index=int(get_val('Bank', 0)))
                    pay = st.selectbox("Pay?", [0, 1], index=int(get_val('Pay', 0)))
                    crypto = st.selectbox("Crypto?", [0, 1], index=int(get_val('Crypto', 0)))
                    has_copyright = st.selectbox("Has Copyright Info?", [0, 1], index=int(get_val('HasCopyrightInfo', 1)))
                    no_image = st.number_input("No of Image", min_value=0, value=int(get_val('NoOfImage', 5)), step=1)
                    no_css = st.number_input("No of CSS", min_value=0, value=int(get_val('NoOfCSS', 3)), step=1)
                
                col6, col7 = st.columns(2)
                with col6:
                    no_js = st.number_input("No of JS", min_value=0, value=int(get_val('NoOfJS', 5)), step=1)
                    no_self_ref = st.number_input("No of Self Ref", min_value=0, value=int(get_val('NoOfSelfRef', 10)), step=1)
                with col7:
                    no_empty_ref = st.number_input("No of Empty Ref", min_value=0, value=int(get_val('NoOfEmptyRef', 1)), step=1)
                    no_ext_ref = st.number_input("No of External Ref", min_value=0, value=int(get_val('NoOfExternalRef', 5)), step=1)
            
            submitted = st.form_submit_button("Prediksi", width='stretch')
        
        if submitted:
            input_data = {
                'URLLength': url_length, 'DomainLength': domain_length,
                'IsDomainIP': is_domain_ip, 'URLSimilarityIndex': url_sim_index,
                'CharContinuationRate': char_cont_rate, 'TLDLegitimateProb': tld_legit_prob,
                'URLCharProb': url_char_prob, 'TLDLength': tld_length,
                'NoOfSubDomain': no_subdomain, 'HasObfuscation': has_obfuscation,
                'NoOfObfuscatedChar': no_obfuscated_char, 'ObfuscationRatio': obfuscation_ratio,
                'NoOfLettersInURL': no_letters, 'LetterRatioInURL': letter_ratio,
                'NoOfDegitsInURL': no_digits, 'DegitRatioInURL': digit_ratio,
                'NoOfEqualsInURL': no_equals, 'NoOfQMarkInURL': no_qmark,
                'NoOfAmpersandInURL': no_ampersand, 'NoOfOtherSpecialCharsInURL': no_special,
                'SpacialCharRatioInURL': special_ratio, 'IsHTTPS': is_https,
                'LineOfCode': line_of_code, 'LargestLineLength': largest_line,
                'HasTitle': has_title, 'DomainTitleMatchScore': domain_title_match,
                'URLTitleMatchScore': url_title_match, 'HasFavicon': has_favicon,
                'Robots': robots, 'IsResponsive': is_responsive,
                'NoOfURLRedirect': no_url_redirect, 'NoOfSelfRedirect': no_self_redirect,
                'HasDescription': has_description, 'NoOfPopup': no_popup,
                'NoOfiFrame': no_iframe, 'HasExternalFormSubmit': has_ext_form,
                'HasSocialNet': has_social, 'HasSubmitButton': has_submit,
                'HasHiddenFields': has_hidden, 'HasPasswordField': has_password,
                'Bank': bank, 'Pay': pay, 'Crypto': crypto,
                'HasCopyrightInfo': has_copyright, 'NoOfImage': no_image,
                'NoOfCSS': no_css, 'NoOfJS': no_js,
                'NoOfSelfRef': no_self_ref, 'NoOfEmptyRef': no_empty_ref,
                'NoOfExternalRef': no_ext_ref
            }
            
            try:
                df_input = pd.DataFrame([input_data])
                available_features = [c for c in feature_names if c in df_input.columns]
                df_features = df_input[available_features]
                df_scaled = pd.DataFrame(scaler.transform(df_features), columns=available_features)
                df_selected = df_scaled[[c for c in selected_features if c in df_scaled.columns]]
                
                pred = model.predict(df_selected)
                
                st.markdown("---")
                
                if pred[0] == 1:
                    st.error("Hasil Prediksi: PHISHING TERDETEKSI")
                else:
                    st.success("Hasil Prediksi: URL AMAN (LEGITIMATE)")
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(df_selected)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Probabilitas Aman", f"{proba[0][0]:.2%}")
                    with col2:
                        st.metric("Probabilitas Phishing", f"{proba[0][1]:.2%}")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba[0][1] * 100,
                        title={'text': "Persentase Risiko Phishing"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': '#e74c3c' if proba[0][1] > 0.5 else '#2ecc71'},
                            'steps': [
                                {'range': [0, 30], 'color': '#f2f9f2'},
                                {'range': [30, 70], 'color': '#fef7ec'},
                                {'range': [70, 100], 'color': '#fdf2f2'}
                            ]
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, width='stretch')
                
                st.warning("Catatan: Hasil prediksi ini adalah pendukung keputusan, bukan keputusan final keamanan siber.")
                
            except Exception as e:
                st.error(f"Error saat memproses data: {str(e)}")


# =============================================================================
# PAGE 3: PERBANDINGAN MODEL
# =============================================================================
elif page == "Perbandingan Model":
    st.title("Perbandingan Performa Model")
    st.subheader("Evaluasi Metrik Baseline vs Optimized (KNN, Naive Bayes, SVM)")
    st.markdown("---")
    
    if reports.get('experiments') is not None:
        df_exp = reports['experiments']
        
        st.subheader("Tabel Hasil Eksperimen")
        df_display = df_exp.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Macro-F1', 'Balanced Acc']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        st.dataframe(df_display, width='stretch')
        
        st.markdown("---")
        
        st.subheader("Grafik Perbandingan Metrik")
        metric_choice = st.selectbox(
            "Pilih Metrik Evaluasi:", 
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Macro-F1', 'Balanced Acc']
        )
        
        fig = px.bar(
            df_exp, x='Model', y=metric_choice, color='Type',
            barmode='group',
            color_discrete_map={'Baseline': '#3498db', 'Optimized': '#9b59b6'},
            text=metric_choice
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(height=400, yaxis_range=[0, 1.1])
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        st.subheader("Radar Chart Model Teroptimasi")
        
        optimized = df_exp[df_exp['Type'] == 'Optimized']
        metrics_radar = ['Accuracy', 'Precision', 'Recall', 'Macro-F1', 'Balanced Acc']
        
        fig = go.Figure()
        colors = {'KNN': '#2980b9', 'Naive Bayes': '#27ae60', 'SVM': '#d35400'}
        
        for _, row in optimized.iterrows():
            values = [row[m] for m in metrics_radar]
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_radar + [metrics_radar[0]],
                fill='toself',
                name=row['Model'],
                line=dict(color=colors.get(row['Model'], '#7f8c8d'))
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("Data hasil eksperimen tidak ditemukan. Jalankan pipeline training terlebih dahulu.")
    
    # Confusion Matrix
    if reports.get('errors'):
        st.markdown("---")
        st.subheader("Confusion Matrix Model Teroptimasi")
        
        cols = st.columns(3)
        for i, (name, ea) in enumerate(reports['errors'].items()):
            with cols[i]:
                cm = np.array([
                    [ea['true_negatives'], ea['false_positives']],
                    [ea['false_negatives'], ea['true_positives']]
                ])
                
                fig = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    x=['Legitimate', 'Phishing'],
                    y=['Legitimate', 'Phishing'],
                    color_continuous_scale='Blues'
                )
                fig.update_layout(title=f"{name}", height=300)
                st.plotly_chart(fig, width='stretch')


# =============================================================================
# PAGE 4: VISUALISASI
# =============================================================================
elif page == "Visualisasi":
    st.title("Visualisasi & Analisis Fitur")
    st.markdown("---")
    
    df_sample = load_dataset_sample()
    
    if df_sample is not None:
        tab1, tab2, tab3 = st.tabs(["Distribusi Fitur", "Korelasi Fitur", "Analisis Kesalahan"])
        
        with tab1:
            st.subheader("Distribusi Fitur Berdasarkan Label Kelas")
            numeric_features = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            if 'label' in numeric_features:
                numeric_features.remove('label')
            
            selected = st.selectbox("Pilih Fitur untuk Boxplot:", numeric_features[:20])
            
            fig = px.box(
                df_sample, x='label', y=selected,
                color='label',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'label': 'Label Kelas'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with tab2:
            st.subheader("Matriks Korelasi (Top 15 Fitur Terkorelasi)")
            cols_to_exclude = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
            df_numeric = df_sample.drop(columns=[c for c in cols_to_exclude if c in df_sample.columns], errors='ignore')
            
            corr_with_label = df_numeric.corr()['label'].abs().sort_values(ascending=False)
            top_features = corr_with_label.head(16).index.tolist()
            
            corr_matrix = df_numeric[top_features].corr()
            
            fig = px.imshow(
                corr_matrix, text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader("Analisis Kesalahan Prediksi (Error Analysis)")
            if reports.get('errors'):
                for name, ea in reports['errors'].items():
                    with st.expander(f"Model: {name}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Error", ea['total_errors'])
                        with col2:
                            st.metric("Error Rate", f"{ea['error_rate']}%")
                        with col3:
                            st.metric("Total Sampel Uji", ea['total_samples'])
                        
                        col4, col5 = st.columns(2)
                        with col4:
                            st.metric("False Positives", ea['false_positives'])
                        with col5:
                            st.metric("False Negatives", ea['false_negatives'])
                        
                        if ea.get('top_differentiating_features'):
                            st.markdown("**Perbedaan Rata-rata Fitur pada Salah Prediksi:**")
                            feat_df = pd.DataFrame(
                                list(ea['top_differentiating_features'].items()),
                                columns=['Fitur', 'Selisih Rata-rata']
                            )
                            st.dataframe(feat_df, width='stretch')
            else:
                st.info("Laporan analisis kesalahan tidak ditemukan.")


# =============================================================================
# PAGE 5: TENTANG
# =============================================================================
elif page == "Tentang":
    st.title("Tentang Proyek")
    st.markdown("---")
    
    st.markdown("""
    ### Sistem Klasifikasi Deteksi Phishing URL
    
    Aplikasi ini dibangun untuk melakukan prediksi keamanan siber secara mandiri guna mengidentifikasi URL yang berpotensi menjadi ancaman phishing (pencurian data).
    
    ### Metodologi Eksperimen
    
    1. **Dataset**: Dataset PhiUSIIL Phishing URL dari repositori pembelajaran mesin UCI (235.795 data).
    2. **Algoritma Wajib**: Pembandingan KNN, Gaussian Naive Bayes, dan Support Vector Machine (SVM).
    3. **Optimasi**: Penggunaan SMOTE untuk mengatasi ketidakseimbangan kelas, GridSearchCV untuk pencarian parameter terbaik, serta validasi silang Stratified 5-Fold.
    4. **Seleksi Fitur**: Pemeringkatan fitur menggunakan skor Mutual Information.
    
    ### Aspek Etika dan Batasan
    
    - Model yang dibangun murni ditujukan sebagai alat bantu pengambilan keputusan (*decision support*), bukan sebagai firewall otomatis.
    - Privasi data pengguna tetap terjaga karena aplikasi tidak mengunggah atau menyimpan aktivitas eksplorasi URL ke pangkalan data eksternal.
    - Evaluasi model berfokus pada metrik F1-Score makro untuk menjamin keseimbangan akurasi pada kelas aman maupun bahaya.
    
    ---
    
    ### Informasi Akademik
    - **Nama Mahasiswa**: Mayolus Gavin
    - **NIM**: A11.2024.15574
    - **Mata Kuliah**: Pembelajaran Mesin
    - **Dosen Pengampu**: Junta Zeniarja, M.Kom
   
    """)
    
    if model_info:
        st.markdown("---")
        st.subheader("Spesifikasi Model Terpilih")
        st.json(model_info)
