import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# โหลดโมเดลและ scaler จากไฟล์
with open('model_penguin_65130701938.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_penguin.pkl', 'rb') as f:
    scaler = pickle.load(f)

# สร้างฟังก์ชันสำหรับพยากรณ์
def predict_species(island, culmen_length, culmen_depth, flipper_length, body_mass, sex):
    input_data = np.array([[island, culmen_length, culmen_depth, flipper_length, body_mass, sex]])
    input_data_scaled = scaler.transform(input_data)  # สเกลข้อมูล
    
    prediction = model.predict(input_data_scaled)
    predicted_species = species_encoder.inverse_transform(prediction)
    
    return predicted_species[0]

# สร้าง UI สำหรับการกรอกข้อมูล
st.title('Penguin Species Prediction')

island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length = st.number_input('Culmen Length (mm)', min_value=30.0, max_value=70.0)
culmen_depth = st.number_input('Culmen Depth (mm)', min_value=10.0, max_value=30.0)
flipper_length = st.number_input('Flipper Length (mm)', min_value=150.0, max_value=250.0)
body_mass = st.number_input('Body Mass (g)', min_value=2500.0, max_value=7000.0)
sex = st.selectbox('Sex', ['Male', 'Female'])

# แปลงข้อมูล Island และ Sex เป็นตัวเลข
island_map = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
sex_map = {'Male': 0, 'Female': 1}

# เมื่อผู้ใช้คลิกปุ่ม 'Predict'
if st.button('Predict'):
    species = predict_species(island_map[island], culmen_length, culmen_depth, flipper_length, body_mass, sex_map[sex])
    st.write(f'The predicted species is: {species}')
