import os
import pandas as pd
import joblib
import pdfplumber  # Extract data from PDF
import re

# Load trained models
lc = joblib.load("Diabetespredictionmodel(LC).model")
rfc = joblib.load("Diabetespredictionmodel(RFC).model")

print("Current Working Directory:", os.getcwd())

# Extract key parameters from the uploaded PDF
def extract_values_from_pdf(pdf_path):
    extracted_data = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            # Extract glucose fasting
            match = re.search(r'Glucose fasting \(PHO\)\s+(\d+)\s+mg/dl', text)
            if match:
                extracted_data["Glucose_Fasting"] = float(match.group(1))
  
            # Extract cholesterol levels
            match = re.search(r'Cholesterol, total \(PHO\)\s+(\d+)\s+mg/dl', text)
            if match:
                extracted_data["Serum_Cholesterol"] = float(match.group(1))

            match = re.search(r'Triglycerides \(PHO\)\s+(\d+)\s+mg/dl', text)
            if match:
                extracted_data["Serum_Triglycerides"] = float(match.group(1))

            match = re.search(r'HDL Cholesterol, direct \(PHO\)\s+([\d.]+)\s+mg/dl', text)
            if match:
                extracted_data["HDL_Cholesterol"] = float(match.group(1))

            match = re.search(r'LDL Cholesterol, direct \(PHO\)\s+(\d+)\s+mg/dl', text)
            if match:
                extracted_data["LDL_Cholesterol"] = float(match.group(1))

            # Extract HbA1c
            match = re.search(r'Hb A1c \(TURB\)\s+([\d.]+)\s+%', text)
            if match:
                extracted_data["HbA1c"] = float(match.group(1))

            # Extract Creatinine
            match = re.search(r'Creatinine \(PHO\)\s+([\d.]+)\s+mg/dl', text)
            if match:
                extracted_data["Creatinine"] = float(match.group(1))

    return extracted_data

# Extract patient details and lab results
pdf_path = "fml-diabetes-profile-sample-report.pdf"
patient_data = extract_values_from_pdf(pdf_path)

# Add additional patient info manually
patient_data.update({
    "Age_Group": 50,  # Derived from DOB
    "Gender": 1,  # Assuming 1 = Female, 0 = Male
    "Height_Feet": 5.6,  # Example value
    "Weight_Kg": 75,  # Example value
    "BMI": 24.7,  # Example value
    "Blood_Pressure": "120/80",  # Example value
    "Heart_Rate": 72,  # Example value
})

# Convert data to DataFrame format
patient_df = pd.DataFrame([patient_data])

# Make Predictions
y_pred_lc = lc.predict(patient_df.drop(columns=["Blood_Pressure", "Heart_Rate"]))
y_pred_rfc = rfc.predict(patient_df.drop(columns=["Blood_Pressure", "Heart_Rate"]))

# Store Predictions
patient_df["Diabetes_Prediction_LC"] = y_pred_lc
patient_df["Diabetes_Prediction_RFC"] = y_pred_rfc

# Create Directories for Classification
diabetic_folder = "Diabetic_Patients"
non_diabetic_folder = "Non_Diabetic_Patients"

os.makedirs(diabetic_folder, exist_ok=True)
os.makedirs(non_diabetic_folder, exist_ok=True)

# Save Predictions Separately
if y_pred_lc[0] == 1:
    patient_df.to_csv(os.path.join(diabetic_folder, "patient_report.csv"), index=False)
else:
    patient_df.to_csv(os.path.join(non_diabetic_folder, "patient_report.csv"), index=False)

# Save Full Report
patient_df.to_csv("Diabetes_Prediction_Report.csv", index=False)

# Display results
print("\nClassification Matrix of Logistic Regression:")
print(y_pred_lc)

print("\nClassification Matrix of Random Forest:")
print(y_pred_rfc)

print("\n✅ Processing Complete! Reports saved locally.")

