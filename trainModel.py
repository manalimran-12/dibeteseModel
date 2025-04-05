# import os
# import pandas as pd
# import joblib
# import pdfplumber  # Extract data from PDF
# import re

# # Load trained models
# lc = joblib.load("Diabetespredictionmodel(LC).model")
# rfc = joblib.load("Diabetespredictionmodel(RFC).model")

# print("Current Working Directory:", os.getcwd())

# # Extract key parameters from the uploaded PDF
# def extract_values_from_pdf(pdf_path):
#     extracted_data = {}

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()

#             # Extract glucose fasting
#             match = re.search(r'Glucose fasting \(PHO\)\s+(\d+)\s+mg/dl', text)
#             if match:
#                 extracted_data["Glucose_Fasting"] = float(match.group(1))
  
#             # Extract cholesterol levels
#             match = re.search(r'Cholesterol, total \(PHO\)\s+(\d+)\s+mg/dl', text)
#             if match:
#                 extracted_data["Serum_Cholesterol"] = float(match.group(1))

#             match = re.search(r'Triglycerides \(PHO\)\s+(\d+)\s+mg/dl', text)
#             if match:
#                 extracted_data["Serum_Triglycerides"] = float(match.group(1))

#             match = re.search(r'HDL Cholesterol, direct \(PHO\)\s+([\d.]+)\s+mg/dl', text)
#             if match:
#                 extracted_data["HDL_Cholesterol"] = float(match.group(1))

#             match = re.search(r'LDL Cholesterol, direct \(PHO\)\s+(\d+)\s+mg/dl', text)
#             if match:
#                 extracted_data["LDL_Cholesterol"] = float(match.group(1))

#             # Extract HbA1c
#             match = re.search(r'Hb A1c \(TURB\)\s+([\d.]+)\s+%', text)
#             if match:
#                 extracted_data["HbA1c"] = float(match.group(1))

#             # Extract Creatinine
#             match = re.search(r'Creatinine \(PHO\)\s+([\d.]+)\s+mg/dl', text)
#             if match:
#                 extracted_data["Creatinine"] = float(match.group(1))

#     return extracted_data

# # Extract patient details and lab results
# pdf_path = "fml-diabetes-profile-sample-report.pdf"
# patient_data = extract_values_from_pdf(pdf_path)

# # Add additional patient info manually
# patient_data.update({
#     "Age_Group": 50,  # Derived from DOB
#     "Gender": 1,  # Assuming 1 = Female, 0 = Male
#     "Height_Feet": 5.6,  # Example value
#     "Weight_Kg": 75,  # Example value
#     "BMI": 24.7,  # Example value
#     "Blood_Pressure": "120/80",  # Example value
#     "Heart_Rate": 72,  # Example value
# })

# # Convert data to DataFrame format
# patient_df = pd.DataFrame([patient_data])

# # Make Predictions
# y_pred_lc = lc.predict(patient_df.drop(columns=["Blood_Pressure", "Heart_Rate"]))
# y_pred_rfc = rfc.predict(patient_df.drop(columns=["Blood_Pressure", "Heart_Rate"]))

# # Store Predictions
# patient_df["Diabetes_Prediction_LC"] = y_pred_lc
# patient_df["Diabetes_Prediction_RFC"] = y_pred_rfc

# # Create Directories for Classification
# diabetic_folder = "Diabetic_Patients"
# non_diabetic_folder = "Non_Diabetic_Patients"

# os.makedirs(diabetic_folder, exist_ok=True)
# os.makedirs(non_diabetic_folder, exist_ok=True)

# # Save Predictions Separately
# if y_pred_lc[0] == 1:
#     patient_df.to_csv(os.path.join(diabetic_folder, "patient_report.csv"), index=False)
# else:
#     patient_df.to_csv(os.path.join(non_diabetic_folder, "patient_report.csv"), index=False)

# # Save Full Report
# patient_df.to_csv("Diabetes_Prediction_Report.csv", index=False)

# # Display results
# print("\nClassification Matrix of Logistic Regression:")
# print(y_pred_lc)

# print("\nClassification Matrix of Random Forest:")
# print(y_pred_rfc)

# print("\n✅ Processing Complete! Reports saved locally.")

# import os
# import pandas as pd
# import joblib
# import pdfplumber
# import re
# from PIL import Image
# import pytesseract

# # Set path to Tesseract (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Load ML models
# lc = joblib.load("Diabetespredictionmodel(LC).model")
# rfc = joblib.load("Diabetespredictionmodel(RFC).model")

# # OCR from image
# def extract_text_from_image(image_path):
#     img = Image.open(image_path)
#     return pytesseract.image_to_string(img)

# # Extract lab parameters using regex
# def extract_values_from_text(text):
#     extracted_data = {}
#     patterns = {
#         "Glucose_Fasting": r'Glucose fasting.*?(\d+)',
#         "Serum_Cholesterol": r'Cholesterol, total.*?(\d+)',
#         "Serum_Triglycerides": r'Triglycerides.*?(\d+)',
#         "HDL_Cholesterol": r'HDL Cholesterol.*?([\d.]+)',
#         "LDL_Cholesterol": r'LDL Cholesterol.*?(\d+)',
#         "HbA1c": r'Hb A1c.*?([\d.,]+)',
#         "Creatinine": r'Creatinine.*?([\d.]+)'
#     }

#     for key, pattern in patterns.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             value = match.group(1).replace(",", ".")
#             try:
#                 extracted_data[key] = float(value)
#             except:
#                 continue
#     return extracted_data

# # Extract from PDF or image
# def extract_values(file_path):
#     if file_path.lower().endswith(".pdf"):
#         with pdfplumber.open(file_path) as pdf:
#             text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#     else:
#         text = extract_text_from_image(file_path)
#     return extract_values_from_text(text)

# # Ask for missing fields
# def collect_missing_inputs(existing_data):
#     questions = {
#         "Age_Group": "Enter your age (in years): ",
#         "Gender": "Enter your gender (0 = Male, 1 = Female): ",
#         "Height_Feet": "Enter your height in feet (e.g., 5.8): ",
#         "Weight_Kg": "Enter your weight in kg: ",
#         "BMI": "Enter your BMI (or press Enter to auto-calculate): ",
#         "Blood_Pressure": "Enter your blood pressure (e.g., 120/80): ",
#         "Heart_Rate": "Enter your heart rate (bpm): ",
#         "Blood_Group": "Enter your blood group (e.g., O+): ",
#         "Glucose_Level": "Enter your random glucose level: ",
#         "Blood_Count": "Enter your blood count range (e.g., 80-90): "
#     }

#     for key, question in questions.items():
#         if key not in existing_data or existing_data[key] in [None, ""]:
#             value = input(question)
#             if key == "BMI" and value == "":
#                 try:
#                     height = float(existing_data.get("Height_Feet") or input("Enter height (feet): "))
#                     weight = float(existing_data.get("Weight_Kg") or input("Enter weight (kg): "))
#                     existing_data["BMI"] = round(weight / ((height * 0.3048) ** 2), 2)
#                 except:
#                     existing_data["BMI"] = 0
#             else:
#                 existing_data[key] = float(value) if key not in ["Blood_Pressure", "Blood_Group", "Blood_Count"] else value
#     return existing_data

# # Start
# print("📂 Please provide your lab report file (PDF or image):")
# file_path = input("👉 File Path: ").strip()

# # STEP 1: Extract values
# extracted_data = extract_values(file_path)
# print("\n🔍 Extracted from report:")
# print(extracted_data)

# # STEP 2: Ask user for rest
# full_data = collect_missing_inputs(extracted_data)
# df = pd.DataFrame([full_data])

# # STEP 3: Clean for prediction
# non_model_cols = ["Blood_Pressure", "Heart_Rate", "Blood_Group", "Blood_Count"]
# predict_df = df.drop(columns=[col for col in non_model_cols if col in df.columns])

# model_features = lc.feature_names_in_ if hasattr(lc, "feature_names_in_") else predict_df.columns
# for col in model_features:
#     if col not in predict_df.columns:
#         predict_df[col] = 0
# predict_df = predict_df[model_features]

# print("\n📊 Final input to model:")
# print(predict_df)

# # STEP 4: Predict
# df["Diabetes_Prediction_LC"] = lc.predict(predict_df)
# df["Diabetes_Prediction_RFC"] = rfc.predict(predict_df)

# # STEP 5: Format result for UI
# output = {
#     "Diabetes_Status": "Diabetic" if df["Diabetes_Prediction_LC"][0] == 1 else "Non-Diabetic",
#     "Vitals": {
#         "Blood Status": str(df["Blood_Pressure"][0]),
#         "Heart Rate": f"{df['Heart_Rate'][0]} bpm",
#         "Blood Count": str(df["Blood_Count"][0]),
#         "Glucose Level": f"{df['Glucose_Level'][0]} ml"
#     }
# }

# # STEP 6: Save reports
# diabetic_folder = "Diabetic_Patients"
# non_diabetic_folder = "Non_Diabetic_Patients"
# os.makedirs(diabetic_folder, exist_ok=True)
# os.makedirs(non_diabetic_folder, exist_ok=True)

# report_path = os.path.join(
#     diabetic_folder if df["Diabetes_Prediction_LC"][0] == 1 else non_diabetic_folder,
#     "patient_report.csv"
# )

# df.to_csv(report_path, index=False)
# df.to_csv("Diabetes_Prediction_Report.csv", index=False)

# # Final Output
# print("\n✅ Prediction Complete!")
# print("🧠 Logistic Regression:", df['Diabetes_Prediction_LC'][0])
# print("🧠 Random Forest:", df['Diabetes_Prediction_RFC'][0])
# print("📊 Dashboard Info:", output)
# print("📁 Report saved to:", report_path)

import os
import pandas as pd
import joblib
import pdfplumber
import re
from PIL import Image
import pytesseract
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set path to Tesseract (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Step 1: Train and save models if needed
def train_and_save_models():
    print("📦 Models not found — training new ones...")

    template_file = "diabetes_training_data.csv"
    if not os.path.exists(template_file):
        columns = [
            "Age_Group", "Gender", "Height_Feet", "Weight_Kg", "BMI",
            "Glucose_Fasting", "Serum_Cholesterol", "Serum_Triglycerides",
            "HDL_Cholesterol", "LDL_Cholesterol", "HbA1c", "Creatinine",
            "Glucose_Level", "Diabetes_Status"
        ]
        pd.DataFrame(columns=columns).to_csv("training_data_template.csv", index=False)
        print("📄 'training_data_template.csv' created.")
        print("⚠️  Fill it with data and rename it to 'diabetes_training_data.csv' if you want to train manually.")
        return

    df = pd.read_csv(template_file)
    if df.empty or "Diabetes_Status" not in df.columns:
        print("⚠️ 'diabetes_training_data.csv' is missing data or target column.")
        return

    X = df.drop(columns=["Diabetes_Status"]).fillna(0)
    y = df["Diabetes_Status"]

    lc = LogisticRegression(max_iter=1000)
    rfc = RandomForestClassifier(n_estimators=100)

    lc.fit(X, y)
    rfc.fit(X, y)

    joblib.dump(lc, "Diabetespredictionmodel(LC).model")
    joblib.dump(rfc, "Diabetespredictionmodel(RFC).model")
    print("✅ Models trained and saved!")

# OCR from image
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# Extract lab values from report
def extract_values_from_text(text):
    extracted_data = {}
    patterns = {
        "Glucose_Fasting": r'Glucose fasting.*?(\d+)',
        "Serum_Cholesterol": r'Cholesterol, total.*?(\d+)',
        "Serum_Triglycerides": r'Triglycerides.*?(\d+)',
        "HDL_Cholesterol": r'HDL Cholesterol.*?([\d.]+)',
        "LDL_Cholesterol": r'LDL Cholesterol.*?(\d+)',
        "HbA1c": r'Hb A1c.*?([\d.,]+)',
        "Creatinine": r'Creatinine.*?([\d.]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", ".")
            try:
                extracted_data[key] = float(value)
            except:
                continue
    return extracted_data

# From file
def extract_values(file_path):
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        text = extract_text_from_image(file_path)
    return extract_values_from_text(text)

# Ask user to fill in vitals
def collect_missing_inputs(existing_data):
    questions = {
        "Age_Group": "Enter your age (in years): ",
        "Gender": "Enter your gender (0 = Male, 1 = Female): ",
        "Height_Feet": "Enter your height (e.g., 5.8): ",
        "Weight_Kg": "Enter your weight in kg: ",
        "BMI": "Enter your BMI (or press Enter to auto-calculate): ",
        "Blood_Pressure": "Enter your blood pressure (e.g., 120/80): ",
        "Heart_Rate": "Enter your heart rate (bpm): ",
        "Blood_Group": "Enter your blood group (e.g., O+): ",
        "Glucose_Level": "Enter your glucose level (mg/dl): ",
        "Blood_Count": "Enter blood count range (e.g., 80-90): "
    }

    for key, question in questions.items():
        if key not in existing_data or existing_data[key] in [None, ""]:
            value = input(question)
            if key == "BMI" and value == "":
                try:
                    h = float(existing_data.get("Height_Feet") or input("Height: "))
                    w = float(existing_data.get("Weight_Kg") or input("Weight: "))
                    existing_data["BMI"] = round(w / ((h * 0.3048) ** 2), 2)
                except:
                    existing_data["BMI"] = 0
            else:
                existing_data[key] = float(value) if key not in ["Blood_Pressure", "Blood_Group", "Blood_Count"] else value
    return existing_data

# --- MAIN FLOW ---

# Step 0: Train models if they don't exist
if not os.path.exists("Diabetespredictionmodel(LC).model") or not os.path.exists("Diabetespredictionmodel(RFC).model"):
    train_and_save_models()

# Step 1: Load models
lc = joblib.load("Diabetespredictionmodel(LC).model")
rfc = joblib.load("Diabetespredictionmodel(RFC).model")

# Step 2: Get file
file_path = input("📂 Enter PDF/image file path: ").strip()
extracted_data = extract_values(file_path)
print("\n🔍 Extracted Values:", extracted_data)

# Step 3: Ask user for vitals
full_data = collect_missing_inputs(extracted_data)
df = pd.DataFrame([full_data])

# Step 4: Predict
non_model_cols = ["Blood_Pressure", "Heart_Rate", "Blood_Group", "Blood_Count"]
predict_df = df.drop(columns=[col for col in non_model_cols if col in df.columns])

model_features = lc.feature_names_in_ if hasattr(lc, "feature_names_in_") else predict_df.columns
for col in model_features:
    if col not in predict_df.columns:
        predict_df[col] = 0
predict_df = predict_df[model_features]

# Step 5: Make predictions
df["Diabetes_Prediction_LC"] = lc.predict(predict_df)
df["Diabetes_Prediction_RFC"] = rfc.predict(predict_df)

# Step 6: Output for UI/dashboard
output = {
    "Diabetes_Status": "Diabetic" if df["Diabetes_Prediction_LC"][0] == 1 else "Non-Diabetic",
    "Vitals": {
        "Blood Status": str(df["Blood_Pressure"][0]),
        "Heart Rate": f"{df['Heart_Rate'][0]} bpm",
        "Blood Count": str(df["Blood_Count"][0]),
        "Glucose Level": f"{df['Glucose_Level'][0]} ml"
    }
}

# Step 7: Save prediction report
folder = "Diabetic_Patients" if df["Diabetes_Prediction_LC"][0] == 1 else "Non_Diabetic_Patients"
os.makedirs(folder, exist_ok=True)
df.to_csv(os.path.join(folder, "patient_report.csv"), index=False)
df.to_csv("Diabetes_Prediction_Report.csv", index=False)

# Step 8: Append to training data
record = df.copy()
record["Diabetes_Status"] = df["Diabetes_Prediction_LC"]

train_features = [
    "Age_Group", "Gender", "Height_Feet", "Weight_Kg", "BMI",
    "Glucose_Fasting", "Serum_Cholesterol", "Serum_Triglycerides",
    "HDL_Cholesterol", "LDL_Cholesterol", "HbA1c", "Creatinine",
    "Glucose_Level", "Diabetes_Status"
]

# Ensure all required columns exist
for col in train_features:
    if col not in record.columns:
        record[col] = None  # or 0

record = record[train_features]
training_file = "diabetes_training_data.csv"

if os.path.exists(training_file):
    record.to_csv(training_file, mode='a', header=False, index=False)
else:
    record.to_csv(training_file, index=False)


print("\n✅ Prediction Complete!")
print("🧠 Logistic Regression:", df['Diabetes_Prediction_LC'][0])
print("🧠 Random Forest:", df['Diabetes_Prediction_RFC'][0])
print("📊 Dashboard Info:", output)
print(f"📁 Report saved to: {folder}/patient_report.csv")
print("📥 Patient data appended to diabetes_training_data.csv ✅")

# Step 9: Retrain model with updated data
def retrain_and_save_model(training_file="diabetes_training_data.csv"):
    if not os.path.exists(training_file):
        print("⚠️ No training data found for retraining.")
        return

    df = pd.read_csv(training_file)
    if df.empty or "Diabetes_Status" not in df.columns:
        print("⚠️ Training data is empty or invalid. Cannot retrain model.")
        return

    df = df.dropna(subset=["Diabetes_Status"])
    df["Diabetes_Status"] = df["Diabetes_Status"].astype(int)

    if df["Diabetes_Status"].nunique() < 2:
        print("⚠️ Cannot retrain — dataset needs at least 2 classes (0 and 1).")
        return

    X = df.drop(columns=["Diabetes_Status"]).fillna(0)
    y = df["Diabetes_Status"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lc = LogisticRegression(max_iter=1000)
    rfc = RandomForestClassifier(n_estimators=100)

    lc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)

    acc_lc = lc.score(X_test, y_test)
    acc_rfc = rfc.score(X_test, y_test)

    print(f"📈 Logistic Regression Accuracy: {acc_lc * 100:.2f}%")
    print(f"🌲 Random Forest Accuracy: {acc_rfc * 100:.2f}%")

    joblib.dump(lc, "Diabetespredictionmodel(LC).model")
    joblib.dump(rfc, "Diabetespredictionmodel(RFC).model")
    print("🔄 Model retrained and saved successfully ✅")


# Call retrain after prediction is saved
retrain_and_save_model()

