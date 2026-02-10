# Load & prepare data
import pandas as pd

df = pd.read_excel("MentalHealthSurvey.xlsx")

# Convert string to integer

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


# Create target 
RISK_FEATURES = ["depression", "anxiety", "isolation", "future_insecurity"]
df["risk_score"] = df[RISK_FEATURES].mean(axis=1)
df["at_risk"] = (df["risk_score"] >= 3.5).astype(int)

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



df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())


X = df[FEATURES]
y = df["at_risk"]

print("Features along with dtypes: ")
print(df[FEATURES].dtypes)
print(df[FEATURES].isnull().sum())

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)


# Train a random forest model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# feature importance
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importances)

importances.plot(kind="bar", title="Feature Importance")
plt.tight_layout()
plt.show()


FINAL_FEATURES = [
    "average_sleep",
    "academic_pressure",
    "financial_concerns",
    "social_relationships",
    "cgpa",
    "campus_discrimination"
]
