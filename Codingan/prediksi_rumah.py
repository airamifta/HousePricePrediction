import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import os
import requests

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="auto"
)

# Fungsi untuk Memuat Model dan Scaler dari GitHub Release
@st.cache_resource
def load_model_and_scaler():
    model_url = "https://github.com/airamifta/HousePricePrediction/releases/download/v1.0.0/model_rumah.pkl"
    scaler_url = "https://github.com/airamifta/HousePricePrediction/releases/download/v1.0.0/scaler_rumah.pkl"

    model_path = "model_rumah.pkl"
    scaler_path = "scaler_rumah.pkl"

    # Download jika belum ada lokal
    if not os.path.exists(model_path):
        r = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(r.content)

    if not os.path.exists(scaler_path):
        r = requests.get(scaler_url)
        with open(scaler_path, 'wb') as f:
            f.write(r.content)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Fungsi untuk Prediksi
def predict_price(model, scaler, input_data):
    selected_features = ['view', 'condition', 'grade']
    input_data[selected_features] = scaler.transform(input_data[selected_features])
    prediction = model.predict(input_data)[0]
    return int(prediction)

# UI Streamlit
st.title("🏡 Prediksi Harga Rumah")
st.write(
    "Masukkan detail properti rumah di bawah ini untuk memprediksi harga estimasi rumah "
    "berdasarkan model machine learning regresi yang telah dilatih."
)

# Muat model dan scaler
model, scaler = load_model_and_scaler()

# Form Input
with st.form("form_prediksi"):
    st.header("Masukkan Detail Rumah Anda")
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("🛏️ Jumlah Kamar Tidur", min_value=0, max_value=40, value=3)
        bathrooms = st.number_input("🛁 Jumlah Kamar Mandi", min_value=0.0, max_value=15.0, value=2.0, step=0.25)
        sqft_living = st.number_input("🏠 Luas Bangunan (sqft)", min_value=10, max_value=100000, value=2000)
        sqft_lot = st.number_input("🌳 Luas Tanah (sqft)", min_value=10, max_value=100000, value=5000)
        floors = st.number_input("🏢 Jumlah Lantai", min_value=1.0, max_value=10.0, value=1.0, step=0.25)
        waterfront = st.selectbox("🌊 Menghadap Laut?", [0, 1])
        view = st.slider("👀 Pemandangan (0-4)", 0, 4, 0)
        condition = st.slider("🧱 Kondisi Rumah (1-5)", 1, 5, 3)
        grade = st.slider("🏗️ Kualitas Bangunan (1-13)", 1, 13, 7)

    with col2:
        sqft_above = st.number_input("⬆️ Luas Lantai Atas", min_value=10, max_value=100000, value=1800)
        sqft_basement = st.number_input("⬇️ Luas Basement", min_value=10, max_value=100000, value=200)
        yr_built = st.number_input("🏗️ Tahun Dibangun", min_value=1800, max_value=2025, value=1990)
        yr_renovated = st.number_input("🔧 Tahun Renovasi (0 jika belum)", min_value=0, max_value=2025, value=0)
        zipcode = st.number_input("📮 Kode Pos", min_value=97000, max_value=99000, value=98178)
        lat = st.number_input("🌐 Latitude", min_value=40.0, max_value=55.0, format="%.6f", value=47.5112)
        long = st.number_input("🌐 Longitude", min_value=-150.0, max_value=-100.0, format="%.6f", value=-122.257)
        sqft_living15 = st.number_input("📏 Luas Rata-rata Rumah Sekitar", min_value=10, max_value=10000, value=1500)
        sqft_lot15 = st.number_input("📏 Luas Rata-rata Tanah Sekitar", min_value=10, max_value=999000, value=4000)

    submit_button = st.form_submit_button(label="🔍 Prediksi Harga", use_container_width=True)

# Logika Prediksi
if submit_button:
    input_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'zipcode': zipcode,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15
    }

    input_df = pd.DataFrame([input_dict])

   try:
        price_usd = predict_price(model, scaler, df)
        price_idr = price_usd * 16380

        st.subheader("💰 Perkiraan Harga Rumah")
        st.metric("Estimasi Harga (USD)", f"${price_usd:,}")
        st.info(f"Jika dikonversi: **Rp{price_idr:,}** (kurs Rp16.380/USD)")

        st.subheader("📍 Lokasi di Peta")
        st.map(pd.DataFrame({'lat': [lat], 'lon': [long]}))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
