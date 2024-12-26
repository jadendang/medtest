import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("data/Healthcare-Diabetes.csv")

X = data.drop(["Outcome", "Id"], axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "models/model.pkl")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


st.title("Diabetes Prediction")

age = st.number_input("Age", min_value=0, max_value=100)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=120.0)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5)


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

if st.button("Predict"):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data)
    st.write("Prediction:", "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected")