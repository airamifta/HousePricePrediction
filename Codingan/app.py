import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")

st.title("🏡 Prediksi Harga Rumah")
st.markdown("Masukkan detail properti untuk memprediksi harga rumah (model regresi).")

# === Input Form ===
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=0, value=3)
        bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=0.0, value=2.0, step=0.25)
        sqft_living = st.number_input("Luas Bangunan (sqft)", min_value=0, value=2000)
        sqft_lot = st.number_input("Luas Tanah (sqft)", min_value=0, value=5000)
        floors = st.number_input("Jumlah Lantai", min_value=0.0, value=1.0, step=0.25)
        waterfront = st.selectbox("Menghadap Laut?", [0, 1])
        view = st.slider("Pemandangan (0-4)", 0, 4, 0)
        condition = st.slider("Kondisi Rumah (1-5)", 1, 5, 3)
        grade = st.slider("Kualitas Bangunan (1-13)", 1, 13, 7)

    with col2:
        sqft_above = st.number_input("Luas Lantai Atas", min_value=0, value=1800)
        sqft_basement = st.number_input("Luas Basement", min_value=0, value=200)
        yr_built = st.number_input("Tahun Dibangun", min_value=1800, max_value=2025, value=1990)
        yr_renovated = st.number_input("Tahun Renovasi (0 jika belum)", min_value=0, max_value=2025, value=0)
        zipcode = st.number_input("Kode Pos", value=98178)
        lat = st.number_input("Latitude", format="%.6f", value=47.5112)
        long = st.number_input("Longitude", format="%.6f", value=-122.257)
        sqft_living15 = st.number_input("Luas Rata-rata Rumah Sekitar", value=1500)
        sqft_lot15 = st.number_input("Luas Rata-rata Tanah Sekitar", value=4000)

    submitted = st.form_submit_button("🔍 Prediksi Harga")

# === Prediksi ===
if submitted:
    try:
        # Susun input menjadi DataFrame
        feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                         'waterfront', 'view', 'condition', 'grade',
                         'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                         'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

        input_df = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                                  waterfront, view, condition, grade,
                                  sqft_above, sqft_basement, yr_built, yr_renovated,
                                  zipcode, lat, long, sqft_living15, sqft_lot15]],
                                columns=feature_names)

        # Load dan terapkan scaler
        scaler = joblib.load("scaler_rumah.pkl")
        input_df[['view', 'condition', 'grade']] = scaler.transform(input_df[['view', 'condition', 'grade']])

        # Load model
        model = joblib.load("model_rumah.pkl")

        # Prediksi
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Perkiraan Harga Rumah: **${int(prediction):,}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
