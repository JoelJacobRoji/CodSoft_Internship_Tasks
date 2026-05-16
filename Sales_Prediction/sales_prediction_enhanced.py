import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox

# ==========================================
# STEP 1: Load Data
# ==========================================
print("\nLoading Advertising dataset...")
try:
    df = pd.read_csv('advertising.csv')
except FileNotFoundError:
    print("ERROR: 'advertising.csv' not found. Please ensure it is in the same folder.")
    exit()

# ==========================================
# STEP 2: Exploratory Data Analysis (EDA)
# ==========================================
print("Generating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Advertising vs. Sales")
plt.show()

# ==========================================
# STEP 3 & 4: Preprocessing & Training
# ==========================================
print("Training the Linear Regression model...")
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# STEP 5: Evaluation & Visual Proof
# ==========================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# ENHANCEMENT 1: Actual vs Predicted Scatter Plot
print("\nGenerating Actual vs Predicted Plot... ")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Model Accuracy: Actual vs. Predicted Sales")
plt.show()

# ENHANCEMENT 2: ROI Coefficient Visual
print("Generating ROI Visual...")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
plt.figure(figsize=(8, 5))
sns.barplot(x=coefficients.index, y=coefficients['Coefficient'], hue=coefficients.index, palette='viridis', legend=False)
plt.title("Return on Investment (ROI) per $1 Spent")
plt.ylabel("Sales Generated per $1")
plt.show()

# ==========================================
# STEP 6: STRATEGIC BUSINESS RECOMMENDATIONS
# ==========================================
print("\n--- Business Insights: The ROI Formula ---\n")
print(coefficients)


print("\n\n📊 STRATEGIC BUSINESS INSIGHTS & RECOMMENDATIONS\n")

print("1. THE EFFICIENCY WINNER (Radio):")
print("   Dollar-for-dollar, Radio is the most efficient platform.")
print("   Every $1 spent yields the highest number of extra sales.")
print("\n2. THE VOLUME DRIVER (TV):")
print("   While less efficient than Radio, TV drives the vast majority")
print("   of total sales simply because the budget allocated to it is massive.")
print("\n3. THE DEAD WEIGHT (Newspaper):")
print("   The Newspaper coefficient is nearly zero. Money spent here is")
print("   having a negligible impact on our final sales volume.")
print("\n💡 ACTIONABLE RECOMMENDATION:")
print("   - IMMEDIATELY cut the Newspaper budget to zero and shift those funds to Radio.")
print("   - DO NOT cut the TV budget drastically. Instead, run an A/B test by")
print("     shifting 10% of the TV budget into Radio next quarter to capitalize")
print("     on its high ROI without hitting market saturation (diminishing returns).")

# ==========================================
# STEP 7: INTERACTIVE BUDGET SIMULATOR (GUI)
# ==========================================
print("Launching Interactive Budget Simulator UI...")

def predict_sales():
    try:
        # Get values from UI
        tv_budget = float(entry_tv.get())
        radio_budget = float(entry_radio.get())
        news_budget = float(entry_news.get())
        
        # Predict
        input_data = pd.DataFrame([[tv_budget, radio_budget, news_budget]], columns=['TV', 'Radio', 'Newspaper'])
        prediction = model.predict(input_data)[0]
        
        # Update UI
        result_label.config(text=f"Predicted Sales: {prediction:.2f} Units", fg="#1565C0")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all budgets.")

# Create main window
root = tk.Tk()
root.title("Marketing Budget Simulator")
root.geometry("400x450")
root.configure(bg="#f4f6f9")

tk.Label(root, text="📈 Sales Prediction Simulator", font=("Helvetica", 16, "bold"), bg="#f4f6f9", fg="#333").pack(pady=20)

frame = tk.Frame(root, bg="#f4f6f9")
frame.pack(pady=10)

def create_input(parent, label, default_val):
    row = tk.Frame(parent, bg="#f4f6f9")
    row.pack(fill="x", pady=5)
    tk.Label(row, text=label, width=15, anchor="w", font=("Helvetica", 11), bg="#f4f6f9").pack(side="left")
    entry = ttk.Entry(row, font=("Helvetica", 11), width=15)
    entry.insert(0, default_val)
    entry.pack(side="right")
    return entry

# Default inputs based on dataset averages
entry_tv = create_input(frame, "TV Budget ($):", "150.0")
entry_radio = create_input(frame, "Radio Budget ($):", "25.0")
entry_news = create_input(frame, "Newspaper ($):", "10.0")

tk.Button(root, text="Predict Sales", font=("Helvetica", 12, "bold"), bg="#2E7D32", fg="white", 
          command=predict_sales, padx=20, pady=5, relief="flat").pack(pady=20)

result_label = tk.Label(root, text="Predicted Sales: -- Units", font=("Helvetica", 16, "bold"), bg="#f4f6f9", fg="#555")
result_label.pack(pady=10)

root.mainloop()