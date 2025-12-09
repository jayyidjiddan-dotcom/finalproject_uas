import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("weather_rf_model.pkl")  # ganti sesuai nama file model kamu

st.title("Weather Temperature Prediction App")
st.write("Aplikasi untuk memprediksi suhu berdasarkan parameter cuaca.")

# Input user
tempmax = st.number_input("Temperatur Maksimum (°C)", 0.0, 60.0, 30.0)
tempmin = st.number_input("Temperatur Minimum (°C)", -20.0, 40.0, 20.0)
humidity = st.number_input("Kelembaban (%)", 0.0, 100.0, 75.0)
windspeed = st.number_input("Kecepatan Angin (km/h)", 0.0, 100.0, 10.0)
sealevelpressure = st.number_input("Tekanan Udara (hPa)", 900.0, 1100.0, 1010.0)

conditions = st.selectbox("Kondisi Cuaca", 
    ["Clear", "Partially cloudy", "Overcast", "Rain"]
)

# Encode conditions sama seperti data training
df = pd.DataFrame({
    "tempmax": [tempmax],
    "tempmin": [tempmin],
    "humidity": [humidity],
    "windspeed": [windspeed],
    "sealevelpressure": [sealevelpressure],
    "conditions": [conditions]
})

df = pd.get_dummies(df, columns=['conditions'], drop_first=True)

# Pastikan kolom dummy sesuai model
for col in model.feature_names_in_:
    if col not in df.columns:
        df[col] = 0  # tambah kolom yang hilang

df = df[model.feature_names_in_]  # urutkan kolom sesuai model

# Prediksi
if st.button("Prediksi Suhu"):
    y_pred = model.predict(df)[0]
    st.success(f"Prediksi Suhu: {y_pred:.2f} °C")
