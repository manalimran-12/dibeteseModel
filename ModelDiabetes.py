#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries: 

# In[1]:


import numpy as np
import os 


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


from imblearn.over_sampling import SMOTE


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[11]:


import joblib as jb


# # Reading data: 

# In[12]:


df = pd.read_csv('Pakistani_Diabetes_Dataset.csv')


# In[13]:


df.head(5)


# # EDA: 

# In[14]:


df.info()


# In[15]:


le = LabelEncoder()


# In[16]:


for i in ['Age_Group', 'Gender', 'Diabetes_Detection', 'Diabetes_Level', 'Sugar_Type']:
    df[i] = le.fit_transform(df[i])


# In[17]:


df.sample(2)


# In[18]:


plt.figure(figsize=(25, 5))
df.corr()['Diabetes_Detection'].sort_values().plot(kind='bar')


# In[19]:


plt.figure(figsize=(25, 5))
sns.heatmap(df.corr(), annot=True, cmap="RdBu")


# In[20]:


# fig = plt.figure(figsize=(25, 5))
# sns.pairplot(df, hue='Diabetes_Detection', diag_kind='kde')


# In[21]:


diabetes_detection = df.pivot_table(columns='Diabetes_Detection', values=['Serum_Cholesterol', 'HbA1c', 'Hemoglobin', 'CGFR', 'BMI'], aggfunc='mean')
diabetes_detection


# In[22]:


diabetes_detection.plot(kind='bar')


# In[23]:


diabetes_aged = df.pivot_table(columns='Age_Group', values=['Serum_Cholesterol', 'HbA1c', 'Hemoglobin', 'CGFR', 'BMI'], aggfunc='mean')
diabetes_aged


# In[24]:


diabetes_aged.plot(kind='bar')


# In[25]:


diabetes_gender = df.pivot_table(columns='Gender', values=['Serum_Cholesterol', 'HbA1c', 'Hemoglobin', 'CGFR', 'BMI'], aggfunc='mean')
diabetes_gender


# In[26]:


diabetes_gender.plot(kind='bar')


# In[27]:


diabetes_sugartype = df.pivot_table(columns='Sugar_Type', values=['LDL_Cholesterol', 'HDL_Cholesterol', 'Hemoglobin', 'CGFR', 'BMI'], aggfunc='mean')
diabetes_sugartype


# In[28]:


diabetes_sugartype.plot(kind='bar')


# # Machine Learning:

# In[29]:


df['Diabetes_Detection'].value_counts(normalize=True)


# In[30]:


X = df.drop('Diabetes_Detection', axis=1)


# In[31]:


Y = df['Diabetes_Detection']


# In[32]:


smote = SMOTE(random_state=42)


# In[33]:


X_resampled, y_resampled = smote.fit_resample(X, Y)


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)


# In[35]:


lc = LogisticRegression(C=0.01, max_iter=10, verbose=1)


# In[36]:


rfc = RandomForestClassifier(n_estimators=7, max_depth=2, random_state=42, verbose=1)


# In[37]:


lc.fit(x_train, y_train)


# In[38]:


rfc.fit(x_train, y_train)


# In[39]:


y_pred_lc = lc.predict(x_test)


# In[40]:


y_pred_rfc = rfc.predict(x_test)


# In[41]:


print('Classification Matrix of Logistic Regression: ')
print(classification_report(y_test, y_pred_lc))


# In[42]:


print('Classification Matrix of Random Forest: ')
print(classification_report(y_test, y_pred_rfc))


# In[43]:


plt.title('Confusion Matrix for Random Forest: ')
sns.heatmap(confusion_matrix(y_test, y_pred_rfc), annot=True, cmap='BuPu')


# In[44]:


plt.title('Confusion Matrix for Logistic Regression: ')
sns.heatmap(confusion_matrix(y_test, y_pred_lc), annot=True, cmap='BuPu')


# In[45]:


accuracy_scores = [
    {'Model': 'Random Forest', 'Accuracy Score': accuracy_score(y_test, y_pred_rfc)},
    {'Model': 'Logistic Regression', 'Accuracy Score': accuracy_score(y_test, y_pred_lc)},
]


# In[46]:


accuracy_df = pd.DataFrame(accuracy_scores).sort_values(by='Accuracy Score', ascending=False)


# In[47]:


accuracy_df


# In[48]:


accuracy_df.plot(kind='bar')


# # Saving Model: 

# In[49]:


jb.dump(rfc, 'Diabetespredictionmodel(RFC).model')


# In[50]:


jb.dump(lc, 'Diabetespredictionmodel(LC).model')


# In[51]:


x_test[:2]


# In[52]:


y_test[:2]


# In[53]:


lc.predict([[0,0,4.0,48.0,67.5,25.5,6.1,114.8,0.62,183.6,41.5,169.0,108.4,33.8,14.2,119.2,1,1]])


# In[54]:


rfc.predict([[0,0,4.0,48.0,67.5,25.5,6.1,114.8,0.62,183.6,41.5,169.0,108.4,33.8,14.2,119.2,1,1]])


# In[55]:
x_test['Diabetes_Prediction_LC'] = y_pred_lc  # Logistic Regression
x_test['Diabetes_Prediction_RFC'] = y_pred_rfc  # Random Forest
x_test['Actual_Diabetes_Status'] = y_test.values

# Now Create Folders for Diabetic & Non-Diabetic Patients
diabetic_folder = "Diabetic_Patients"
non_diabetic_folder = "Non_Diabetic_Patients"

os.makedirs(diabetic_folder, exist_ok=True)
os.makedirs(non_diabetic_folder, exist_ok=True)

# Save Patient Records Separately
diabetic_patients = x_test[x_test['Diabetes_Prediction_LC'] == 1]
non_diabetic_patients = x_test[x_test['Diabetes_Prediction_LC'] == 0]

diabetic_csv = os.path.join(diabetic_folder, "diabetic_patients.csv")
non_diabetic_csv = os.path.join(non_diabetic_folder, "non_diabetic_patients.csv")

diabetic_patients.to_csv(diabetic_csv, index=False)
non_diabetic_patients.to_csv(non_diabetic_csv, index=False)

# Save Full Patient Report
full_report_path = "Diabetes_Prediction_Report.csv"
x_test.to_csv(full_report_path, index=False)

# Print classification matrices
print("\nClassification Matrix of Logistic Regression:")
print(classification_report(y_test, y_pred_lc))

print("\nClassification Matrix of Random Forest:")
print(classification_report(y_test, y_pred_rfc))

print("\n✅ Processing Complete! Reports and models saved locally.")

# Done

