import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import joblib

# Global variables
final_model = None
selected_features = None
X_test_selected = None
y_test = None


def train_model():
    global final_model, selected_features, X_test_selected, y_test

    try:
        # Load Dataset
        data = pd.read_csv("synthetic_cvd_data_with_fbs_bp.csv")

        # Preprocess Data
        categorical_features = [
            "Gender",
            "Smoking_Status",
            "Physical_Activity",
            "Alcohol_Consumption",
            "Stress_Levels",
        ]
        encoders = {}
        for column in categorical_features:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            encoders[column] = encoder

        joblib.dump(encoders, "categorical_encoders.pkl")

        numerical_features = [
            "Age",
            "Cholesterol",
            "Systolic_Blood_Pressure",
            "Diastolic_Blood_Pressure",
            "Max_Heart_Rate",
            "BMI",
            "Fasting_Blood_Sugar",
        ]
        scaler = MinMaxScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        joblib.dump(scaler, "scaler.pkl")

        X = data.drop("CVD_Presence", axis=1)
        y = data["CVD_Presence"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Feature Selection with MOBA (simplified)
        num_bats = 20
        max_iterations = 50
        num_features = X_train.shape[1]
        positions = np.random.randint(0, 2, (num_bats, num_features))
        global_best_solution = positions[0]

        def objective_function(solution):
            selected_features = X_train.columns[solution == 1]
            if len(selected_features) == 0:
                return 0, float("inf")
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train[selected_features], y_train)
            y_pred = model.predict(X_test[selected_features])
            return accuracy_score(y_test, y_pred), len(selected_features)

        for _ in range(max_iterations):
            for i in range(num_bats):
                positions[i] = (np.random.random(num_features) > 0.5).astype(int)
                fitness = objective_function(positions[i])
                if fitness[0] > objective_function(global_best_solution)[0]:
                    global_best_solution = positions[i]

        selected_features = X_train.columns[global_best_solution == 1]
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # Train Final Model
        final_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        final_model.fit(X_train_selected, y_train)

        y_pred = final_model.predict(X_test_selected)

        # Save model and features
        joblib.dump(final_model, "xgb_moba_model.pkl")
        joblib.dump(selected_features, "selected_features.pkl")

        accuracy = accuracy_score(y_test, y_pred)

        # Generate Graphs
        generate_graphs(y_test, y_pred, final_model, X_test_selected)

        messagebox.showinfo(
            "Training Completed",
            f"Model trained successfully!\nAccuracy: {accuracy:.2f}",
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def generate_graphs(y_test, y_pred, model, X_test_selected):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_selected)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()


def calculate_bmi():
    try:
        height = float(entries["Height (cm)"].get()) / 100  # Convert to meters
        weight = float(entries["Weight (kg)"].get())
        bmi = weight / (height ** 2)
        entries["BMI"].delete(0, tk.END)
        entries["BMI"].insert(0, f"{bmi:.1f}")
    except ValueError:
        messagebox.showerror("Error", "Invalid Height or Weight")


def predict_cvd():
    try:
        model = joblib.load("xgb_moba_model.pkl")
        selected_features = joblib.load("selected_features.pkl")
        encoders = joblib.load("categorical_encoders.pkl")
        scaler = joblib.load("scaler.pkl")

        input_data = []

        for feature in selected_features:
            if feature in encoders:
                value = entries[feature].get()
                value = encoders[feature].transform([value])[0]
                input_data.append(value)
            else:
                value = float(entries[feature].get())
                scaled_value = scaler.transform(
                    [[value] + [0] * (len(scaler.scale_) - 1)]
                )[0][0]
                input_data.append(scaled_value)

        input_df = pd.DataFrame([input_data], columns=selected_features)
        prediction = model.predict(input_df)
        result = "Presence" if prediction[0] == 1 else "Absence"
        result_label.config(text=f"Predicted CVD Status: {result}", fg="blue", font=("Arial", 16))
        load_and_display_graph("confusion_matrix.png", confusion_matrix_canvas)
        load_and_display_graph("roc_curve.png", roc_curve_canvas)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


def load_and_display_graph(image_path, canvas):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.LANCZOS)  # Updated resizing method
    photo = ImageTk.PhotoImage(img)
    canvas.config(image=photo)
    canvas.image = photo


# GUI Setup
root = tk.Tk()
root.title("CVD Prediction System with Enhanced Layout")
root.geometry("1200x800")

entries = {}

# Left Frame for Input Fields
left_frame = tk.Frame(root, width=400, padx=20, pady=20)
left_frame.pack(side="left", fill="y")

tk.Label(left_frame, text="Enter Values for Prediction", font=("Arial", 14)).pack(pady=10)

field_descriptions = {
    "Age": ("years", "e.g., 45"),
    "Gender": ("options", ["Male", "Female"]),
    "Smoking_Status": ("options", ["Never", "Former", "Current"]),
    "Cholesterol": ("mg/dL", "e.g., 200"),
    "Systolic_Blood_Pressure": ("mmHg", "e.g., 120"),
    "Diastolic_Blood_Pressure": ("mmHg", "e.g., 80"),
    "Fasting_Blood_Sugar": ("mg/dL", "e.g., 90"),
    "Max_Heart_Rate": ("bpm", "e.g., 150"),
    "Height (cm)": ("cm", "e.g., 170"),
    "Weight (kg)": ("kg", "e.g., 70"),
    "BMI": ("kg/m²", "Calculated Automatically"),
    "Physical_Activity": ("options", ["Low", "Moderate", "High"]),
    "Alcohol_Consumption": ("options", ["None", "Occasional", "Frequent"]),
    "Stress_Levels": ("options", ["Low", "Moderate", "High"]),
}

for feature, (unit, example) in field_descriptions.items():
    frame = tk.Frame(left_frame)
    frame.pack(pady=5, anchor="w")
    tk.Label(frame, text=f"{feature} ({unit})", width=30, anchor="w").pack(side="left")
    if isinstance(example, list):
        entry = ttk.Combobox(frame, values=example, width=20)
        entry.set(example[0])
    else:
        entry = tk.Entry(frame, justify="left", width=20)
    entries[feature] = entry
    entry.pack(side="left")

# BMI Calculation Button
tk.Button(left_frame, text="Calculate BMI", command=calculate_bmi, bg="orange", fg="black").pack(pady=10)
tk.Button(left_frame, text="Train Model", command=train_model, bg="green", fg="white").pack(pady=10)
tk.Button(left_frame, text="Predict CVD", command=predict_cvd, bg="blue", fg="white").pack(pady=10)

# Right Frame for Results and Graphs
right_frame = tk.Frame(root, padx=20, pady=20)
right_frame.pack(side="right", fill="both", expand=True)

result_label = tk.Label(right_frame, text="", font=("Arial", 16), fg="blue")
result_label.pack(pady=20)

confusion_matrix_canvas = tk.Label(right_frame)
confusion_matrix_canvas.pack(pady=10)

roc_curve_canvas = tk.Label(right_frame)
roc_curve_canvas.pack(pady=10)

root.mainloop()
