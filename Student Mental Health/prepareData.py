
'''Used only to explore and validate the derived target variable. Not used in training or deployment.'''

import pandas as pd

df = pd.read_excel("MentalHealthSurvey.xlsx")

RISK_FEATURES = [
    "depression",
    "anxiety",
    "isolation",
    "future_insecurity"
]

df["risk_score"] = df[RISK_FEATURES].mean(axis=1)

df["at_risk"] = (df["risk_score"] >= 3.5).astype(int)

print(df["at_risk"].value_counts())
print(df["at_risk"].value_counts(normalize=True))
