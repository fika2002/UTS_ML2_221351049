import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# ========================
# Load TFLite Model & Scaler
# ========================
interpreter = tf.lite.Interpreter(model_path="flight_price_model.tflite")
interpreter.allocate_tensors()

# Ambil detail tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# ========================
# UI
# ========================
st.title("Prediksi Harga Tiket Pesawat INDIA")

airline = st.selectbox("Maskapai", ["AirAsia", "IndiGo", "SpiceJet", "Vistara", "GoAir"])
source_city = st.selectbox("Kota Asal", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"])
destination_city = st.selectbox("Kota Tujuan", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"])
departure_time = st.selectbox("Waktu Keberangkatan", ["Pagi", "Siang", "Sore", "Malam"])
arrival_time = st.selectbox("Waktu Kedatangan", ["Pagi", "Siang", "Sore", "Malam"])
duration = st.number_input("Durasi Penerbangan (menit)", min_value=60, max_value=1000, step=10)
class_type = st.selectbox("Kelas", ["Business","Economy"])
days_left = st.slider("Sisa Hari Keberangkatan", 1, 60)

# ========================
# Manual Encoding
# ========================
airline_dict = {"AirAsia": 0, "IndiGo": 1, "SpiceJet": 2, "Vistara": 3, "GoAir": 4}
city_dict = {"Delhi": 0, "Mumbai": 1, "Bangalore": 2, "Kolkata": 3, "Hyderabad": 4}
time_dict = {"Pagi": 0, "Siang": 1, "Sore": 2, "Malam": 3}
class_dict = {"Business": 0,"Economy": 1}

input_data = np.array([[airline_dict[airline],
                        city_dict[source_city],
                        city_dict[destination_city],
                        time_dict[departure_time],
                        time_dict[arrival_time],
                        duration,
                        class_dict[class_type],
                        days_left]])

# Scaling
input_scaled = scaler.transform(input_data).astype(np.float32)

# ========================
# Prediksi TFLite
# ========================
if st.button("Prediksi"):
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = output[0][0]
    
    kurs_inr_to_idr = 191
    pred_idr = prediction * kurs_inr_to_idr

    st.success(f"Perkiraan Harga Tiket:\n₹{prediction:,.2f} /Rp {pred_idr:,.0f}")
