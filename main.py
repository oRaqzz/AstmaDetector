from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("model/asthma_prediction_model.pkl")

app = FastAPI()

class Patient(BaseModel):
    Age: int
    Gender: str
    BMI: float
    Smoking_Status: str
    Family_History: int
    Allergies: str
    Air_Pollution_Level: str
    Physical_Activity_Level: str
    Occupation_Type: str
    Comorbidities: str
    Medication_Adherence: float
    Number_of_ER_Visits: int
    Peak_Expiratory_Flow: float
    FeNO_Level: float

# Prediction endpoint
@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.dict()])

    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1, 'Other':2})

    smoking_dummies = pd.get_dummies(df['Smoking_Status'], prefix='Smoking')
    df = pd.concat([df, smoking_dummies], axis=1)
    df.drop(columns=['Smoking_Status'], inplace=True)

    df['Air_Pollution_Level'] = df['Air_Pollution_Level'].str.strip().str.title()
    df['Air_Pollution_Level'] = df['Air_Pollution_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

    df['Physical_Activity_Level'] = df['Physical_Activity_Level'].str.strip().str.title()
    df['Physical_Activity_Level'] = df['Physical_Activity_Level'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2})

    df['Occupation_Type'] = df['Occupation_Type'].str.strip().str.title()
    df['Occupation_Type'] = df['Occupation_Type'].map({'Indoor': 0, 'Outdoor': 1})

    comorb_dummies = pd.get_dummies(df['Comorbidities'], prefix='Comorb')
    df = pd.concat([df, comorb_dummies], axis=1)
    df.drop(columns=['Comorbidities'], inplace=True)

    expected_cols = joblib.load("model/model_columns.pkl")
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Predict
    prediction = model.predict(df)[0]
    return {"prediction": "Has Asthma" if prediction == 1 else "No Asthma"}
