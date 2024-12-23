import streamlit as st
import pandas as pd
import plotly.express as px


def main():
    st.title("Diabetes data visualization")

    patient_data = pd.read_csv("data/Healthcare-Diabetes.csv")

    st.write("Patient data")
    st.write(patient_data)

    min_age = st.sidebar.slider("Minimum Age", int(patient_data["Age"].min()), int(patient_data["Age"].max()), int(patient_data["Age"].min()))
    max_age = st.sidebar.slider("Maximum Age", int(patient_data["Age"].min()), int(patient_data["Age"].max()), int(patient_data["Age"].max()))

    gender = st.sidebar.selectbox("Gender", ["All", "Male", "Female"])

    filtered_data = patient_data[(patient_data["Age"] >= min_age) & (patient_data["Age"] <= max_age)]
    if gender != "All":
        filtered_data = filtered_data[filtered_data["Gender"] == gender]

    st.write("Filtered Patient Data:")
    st.write(filtered_data)

    agg_data = filtered_data.groupby("Age")["BloodPressure"].mean().reset_index()

    fig = px.bar(agg_data, x="Age", y="BloodPressure", text="BloodPressure", labels={"BloodPressure": "Average Blood Pressure"}, title="Average Blood Pressure by Age", height=400)

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()