import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STEP 1: Load Data
# ==========================================
print("\n\nLoading Advertising dataset...")
# Load the CSV file
df = pd.read_csv('advertising.csv')

# ==========================================
# STEP 2: Exploratory Data Analysis (EDA)
# ==========================================
print("Generating Correlation Heatmap...")

# We use a heatmap to see the correlation between spending and Sales
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Advertising vs. Sales")
plt.show()

# ==========================================
# STEP 3: Data Preprocessing
# ==========================================
print("Preprocessing data...")
# X = Our Features (The Budgets), y = Our Target (Sales)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: Model Training
# ==========================================
print("Training the Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# STEP 5: Model Evaluation
# ==========================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# ==========================================
# STEP 6: The "Glass Box" (Coefficients)
# ==========================================
print("\n--- Business Insights: The ROI Formula ---\n")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# ==========================================
# STEP 7: STRATEGIC BUSINESS RECOMMENDATIONS
# ==========================================
# This block prints the professional analysis directly to the terminal!

print("\n📊 STRATEGIC BUSINESS INSIGHTS & RECOMMENDATIONS\n")

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
