import numpy as np
import pandas as pd

# Number of samples
num_samples = 50000

# Generate features
age = np.random.randint(30, 80, num_samples)
gender = np.random.choice(['Male', 'Female'], num_samples, p=[0.5, 0.5])
smoking_status = np.random.choice(['Never', 'Former', 'Current'], num_samples, p=[0.4, 0.3, 0.3])
cholesterol = np.random.normal(200, 30, num_samples).astype(int)
systolic_bp = np.random.normal(120, 15, num_samples).astype(int)  # Systolic Blood Pressure
diastolic_bp = np.random.normal(80, 10, num_samples).astype(int)  # Diastolic Blood Pressure
fasting_blood_sugar = np.random.normal(100, 15, num_samples).astype(int)  # Fasting Blood Sugar in mg/dL
max_heart_rate = np.random.normal(150, 20, num_samples).astype(int)
bmi = np.random.normal(25, 5, num_samples).round(1)
physical_activity = np.random.choice(['Low', 'Moderate', 'High'], num_samples, p=[0.3, 0.5, 0.2])
alcohol_consumption = np.random.choice(['None', 'Occasional', 'Frequent'], num_samples, p=[0.5, 0.4, 0.1])
stress_levels = np.random.choice(['Low', 'Moderate', 'High'], num_samples, p=[0.4, 0.4, 0.2])

# Correlate features with CVD presence
cvd_presence = (
    (age > 50).astype(int) +
    (cholesterol > 220).astype(int) +
    (systolic_bp > 140).astype(int) +
    (fasting_blood_sugar > 126).astype(int) +
    (bmi > 30).astype(int)
) > 2

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Smoking_Status': smoking_status,
    'Cholesterol': cholesterol,
    'Systolic_Blood_Pressure': systolic_bp,
    'Diastolic_Blood_Pressure': diastolic_bp,
    'Fasting_Blood_Sugar': fasting_blood_sugar,
    'Max_Heart_Rate': max_heart_rate,
    'BMI': bmi,
    'Physical_Activity': physical_activity,
    'Alcohol_Consumption': alcohol_consumption,
    'Stress_Levels': stress_levels,
    'CVD_Presence': cvd_presence.astype(int)
})

# Save the dataset
data.to_csv('synthetic_cvd_data_with_fbs_bp.csv', index=False)
print("Synthetic dataset saved as 'synthetic_cvd_data_with_fbs_bp.csv'")
