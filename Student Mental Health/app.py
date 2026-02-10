import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Mental Health Risk Predictor", layout="centered")
st.title("Student Mental Health Risk Predictor")
st.write(
    "This application is used to predict the level of mental heath of a student"
)

df = pd.read_excel("MentalHealthSurvey.xlsx")  

# Target creation
RISK_FEATURES = ["depression", "anxiety", "isolation", "future_insecurity"]
df["risk_score"] = df[RISK_FEATURES].mean(axis=1)
df["at_risk"] = (df["risk_score"] >= 3.5).astype(int)

from preprocessing import (
    convert_cgpa,
    convert_sleep,
    convert_sports,
    stress_activity_binary,
    convert_campus_discrimination
)
df["cgpa"] = df["cgpa"].apply(convert_cgpa)
df["average_sleep"] = df["average_sleep"].apply(convert_sleep)
df["sports_engagement"] = df["sports_engagement"].apply(convert_sports)
df["stress_relief_activities"] = df["stress_relief_activities"].apply(stress_activity_binary)
df["campus_discrimination"] = df["campus_discrimination"].apply(convert_campus_discrimination)



# Train model
FEATURES = [
    "age",
    "cgpa",
    "average_sleep",
    "academic_workload",
    "academic_pressure",
    "financial_concerns",
    "social_relationships",
    "campus_discrimination",
    "sports_engagement",
    "study_satisfaction",
    "stress_relief_activities"
]

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df["at_risk"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model

model = train_model(df)


st.subheader("Enter Student Details")

academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
social_relationships = st.slider("Social Relationships", 1, 5, 3)
financial_concerns = st.slider("Financial Concerns", 1, 5, 3)
study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
cgpa = st.slider("CGPA", 2.0, 4.0, 3.0, step=0.1)
academic_workload = st.slider("Academic Workload", 1, 5, 3)

input_df = pd.DataFrame([{
    "academic_pressure": academic_pressure,
    "social_relationships": social_relationships,
    "financial_concerns": financial_concerns,
    "study_satisfaction": study_satisfaction,
    "cgpa": cgpa,
    "academic_workload": academic_workload
}])

# Add missing features with median values
for col in FEATURES:
    if col not in input_df.columns:
        input_df[col] = df[col].median()

input_df = input_df[FEATURES]


if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error("Student is at HIGH RISK")
    else:
        st.success("Student is at LOW RISK")

    st.write(f"**Risk Probability:** {prob:.2f}")


st.subheader("Influence of features on predicting risk")

importances = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

st.bar_chart(importances.head(6))


st.info(
    "This tool does not replace professional mental health diagnosis and is used only for early risk prediction."
)
