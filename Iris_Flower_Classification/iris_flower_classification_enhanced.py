import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk

# ==========================================
# PART 1: DATA PREPARATION & ADVANCED EVALUATION
# ==========================================
print("\n\nLoading data and training model... Please wait.")

# 1. Load Data
df = pd.read_csv('IRIS.csv')

# --- ADDED BACK FROM MVP: Exploratory Data Analysis (EDA) ---
print("\nPairplot Generated !! ")
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette='Dark2', markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features by Species", y=1.02) 
plt.show()

print("\nBoxplots Generated !! ")
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1], 1): 
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=df, hue='species', palette='Set2', legend=False)
    plt.title(f"{feature} by Species")
plt.tight_layout()
plt.show()
# ------------------------------------------------------------

# Separate Features (X) and Target (y)
X = df.drop('species', axis=1)
y = df['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- ENHANCEMENT 1: K-Fold Cross Validation ---
print("\n--- Performing 5-Fold Cross Validation ---")
# This tests the model 5 different times on 5 different slices of the data
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Individual Fold Accuracies: {cv_scores}")
print(f"Average Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%\n")
print("Conclusion: Cross Validation gives more reliable estimate of model's performance across different data.")
print("Reduces the risk of overfitting to single train-test split.")
# --- ADDED BACK FROM MVP: Classification Report & Confusion Matrix ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_test)

print("\n--- Model Accuracy & Classification Report on a single train-test split ---")
print(f"Standard Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix Generated !! ")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title("Confusion Matrix: Iris Species Prediction")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()

# --- ENHANCEMENT 2: Feature Importance ---
print("\nFeature Importance Chart Generated !! ")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
plt.title("Feature Importances: What matters most for Iris Classification?")
plt.bar(range(X.shape[1]), importances[indices], color='mediumseagreen')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices])
plt.ylabel("Importance Score")
plt.show()

# ==========================================
# PART 2: THE GRAPHICAL USER INTERFACE (GUI)
# ==========================================

def predict_species():
    """Reads the slider values and predicts the species."""
    # 1. Get values from sliders
    sl = float(sepal_length_var.get())
    sw = float(sepal_width_var.get())
    pl = float(petal_length_var.get())
    pw = float(petal_width_var.get())
    
    # 2. Format for the model
    input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
    
    # 3. Predict
    prediction = model.predict(input_data)[0]
    
    # 4. Update the UI Text and Color based on species
    color_map = {
        'Iris-setosa': '#e91e63',      # Pink
        'Iris-versicolor': '#4caf50',  # Green
        'Iris-virginica': '#3f51b5'    # Blue
    }
    
    result_label.config(text=f"Predicted Species:\n{prediction}", fg=color_map.get(prediction, "black"))

# --- Build UI Window ---
root = tk.Tk()
root.title("Iris Flower Classifier")
root.geometry("400x550")
root.configure(padx=20, pady=20, bg="#f9f9f9")

title_label = tk.Label(root, text="🌸 Iris Species Predictor", font=("Helvetica", 18, "bold"), bg="#f9f9f9", fg="#333")
title_label.pack(pady=(0, 20))

frame = tk.Frame(root, bg="#f9f9f9")
frame.pack(fill="both", expand=True)

# Helper function to create interactive sliders
def create_slider(parent, label_text, min_val, max_val, default_val):
    row = tk.Frame(parent, bg="#f9f9f9")
    row.pack(fill="x", pady=10)
    
    lbl = tk.Label(row, text=label_text, width=15, anchor="w", bg="#f9f9f9", font=("Helvetica", 10, "bold"))
    lbl.pack(side="left")
    
    # Variable to hold slider value
    val_var = tk.DoubleVar(value=default_val)
    
    # The actual slider widget
    slider = tk.Scale(row, from_=min_val, to=max_val, resolution=0.1, orient="horizontal", 
                      variable=val_var, bg="#f9f9f9", length=180, highlightthickness=0)
    slider.pack(side="right")
    
    return val_var

# Create Sliders based on the actual min/max ranges of the Iris dataset
sepal_length_var = create_slider(frame, "Sepal Length (cm):", 4.0, 8.0, 5.1)
sepal_width_var = create_slider(frame, "Sepal Width (cm):", 2.0, 4.5, 3.5)
petal_length_var = create_slider(frame, "Petal Length (cm):", 1.0, 7.0, 1.4)
petal_width_var = create_slider(frame, "Petal Width (cm):", 0.1, 2.5, 0.2)

predict_btn = tk.Button(root, text="Classify Flower", font=("Helvetica", 12, "bold"), bg="#ff9800", fg="white", 
                        command=predict_species, relief="flat", padx=15, pady=8)
predict_btn.pack(pady=25)

result_label = tk.Label(root, text="Adjust sliders to predict...", font=("Helvetica", 14, "bold"), bg="#f9f9f9", fg="#777")
result_label.pack(pady=10)

# Run the app
root.mainloop()