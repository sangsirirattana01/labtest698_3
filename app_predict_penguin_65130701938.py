
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# โหลดโมเดล
with open("model_penguin_65130701938.pkl", "rb") as file:
    model = pickle.load(file)

# สมมุติว่า scaler ถูกใช้ในการฝึกโมเดล
scaler = StandardScaler()

# รับข้อมูลจากผู้ใช้ผ่าน Streamlit (ตัวอย่าง)
input_data = pd.DataFrame({
    'island': ['Torgersen'],
    'culmen_length_mm': [45.0],
    'culmen_depth_mm': [14.0],
    'flipper_length_mm': [200.0],
    'body_mass_g': [5000.0],
    'sex': ['MALE']
})

# ทำการแปลงข้อมูลให้อยู่ในรูปแบบที่โมเดลคาดหวัง
input_data_scaled = scaler.transform(input_data)

# ทำนายผล
prediction = model.predict(input_data_scaled)

# แสดงผลการทำนาย
st.write("Predicted Species:", prediction)


