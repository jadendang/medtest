import openai
import joblib
import pandas as pd
import streamlit as st

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

openai.api_key = "openai_api_key"

st.title("Diabetes Chatbot")

st.header("Diabetes Prediction")

age = st.number_input("Age", min_value=0, max_value=100)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=120.0)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5)

if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    # Display result
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"
    st.write("Prediction:", result)

st.header("Chatbot")

user_input = st.text_input("Ask GPT a question")

if st.button("Ask GPT"):
    if user_input.strip():
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in healthcare and the dataset of diabetes."},
                {"role": "user", "content": user_input}
            ]
        )
        st.write("GPT:", response.choices[0].message.content)
    else:
        st.write("Please enter a question.")