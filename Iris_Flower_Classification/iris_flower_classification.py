import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# STEP 1: Load Data
# ==========================================
print("Loading Iris dataset...")
# Make sure the file name matches what you downloaded (usually IRIS.csv)
df = pd.read_csv('IRIS.csv')

print("\n--- First 5 rows of the dataset ---")
print(df.head())

# ==========================================
# STEP 2: Exploratory Data Analysis (EDA)
# ==========================================
print("\nGenerating Pairplot... (Close the image window to continue code execution)")

# A Pairplot plots every numerical column against every other numerical column.
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue='species', palette='Dark2', markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features by Species", y=1.02) 
plt.show()

print("\nGenerating Boxplots... (Close the image window to continue code execution)")

# --- NEW: Boxplots for all features ---
plt.figure(figsize=(12, 8))
# We loop through the first 4 columns (the measurements, skipping the species name)
for i, feature in enumerate(df.columns[:-1], 1): 
    plt.subplot(2, 2, i) # Creates a 2x2 grid of graphs
    # Draw the boxplot
    sns.boxplot(x='species', y=feature, data=df, hue='species', palette='Set2', legend=False)
    plt.title(f"{feature} by Species")
    
plt.tight_layout() # Ensures the graphs don't overlap
plt.show()
# --------------------------------------

# ==========================================
# STEP 3: Data Preprocessing
# ==========================================
print("Preprocessing data...")

# Separate features (X) and target label (y)
X = df.drop('species', axis=1) # Everything EXCEPT the species column
y = df['species']              # ONLY the species column

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: Model Training
# ==========================================
print("Training the Random Forest Classifier...")
# Initialize the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# ==========================================
# STEP 5: Model Evaluation
# ==========================================
print("Evaluating model...")

# Predict the species for the test data
y_pred = model.predict(X_test)

# 1. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Accuracy ---")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# 2. Classification Report (Detailed breakdown)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 3. Confusion Matrix Visualization
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title("Confusion Matrix: Iris Species Prediction")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()

# ==========================================
# STEP 6: Testing with a New, Unseen Flower
# ==========================================
print("\n--- Live Prediction Test ---")
# Imagine you found a flower in the wild and measured it:
# Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2
new_flower_measurements = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)

predicted_species = model.predict(new_flower_measurements)
print(f"Measurements: 5.1, 3.5, 1.4, 0.2")
print(f"The model predicts this flower is: {predicted_species[0]}")