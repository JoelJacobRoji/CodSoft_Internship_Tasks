import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ==========================================
# STEP 1: Load Data
# ==========================================
# Load the dataset with latin1 encoding to handle special characters
print("\nLoading data...")
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

# ==========================================
# STEP 2: Data Preprocessing & Cleaning
# ==========================================
print("Cleaning data...")
# Drop missing ratings
df = df.dropna(subset=['Rating'])

# Clean text from numerical columns (ERROR FIXED HERE with .astype(str))
df['Year'] = df['Year'].astype(str).str.extract(r'(\d+)').astype(float)
df['Duration'] = df['Duration'].astype(str).str.replace(' min', '').astype(float)
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(float)

# Fill missing numerical values with median
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# IMPROVEMENT 1: Noise Reduction
# Drop movies with very few votes (less than 50) as their ratings are highly volatile/unreliable
initial_count = len(df)
df = df[df['Votes'] >= 50]
print(f"Dropped {initial_count - len(df)} movies with less than 50 votes to reduce noise.")

# ==========================================
# STEP 3: Exploratory Data Analysis (EDA)
# ==========================================
# IMPROVEMENT 2: Visualize the distribution of the target variable
plt.figure(figsize=(10, 5))
sns.histplot(df['Rating'], bins=30, kde=True, color='purple')
plt.title('Distribution of IMDb Movie Ratings')
plt.xlabel('Rating (1-10)')
plt.ylabel('Number of Movies')
plt.axvline(df['Rating'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["Rating"].mean():.2f}')
plt.legend()
plt.show()

# ==========================================
# STEP 4: Feature Engineering (Target Encoding)
# ==========================================
print("Engineering features...")
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for feature in categorical_features:
    df[feature] = df[feature].fillna('Unknown')
    mean_encodings = df.groupby(feature)['Rating'].mean()
    df[feature + '_encoded'] = df[feature].map(mean_encodings)

features = ['Year', 'Duration', 'Votes', 'Genre_encoded', 'Director_encoded', 
            'Actor 1_encoded', 'Actor 2_encoded', 'Actor 3_encoded']

X = df[features]
y = df['Rating']

# ==========================================
# STEP 5: Model Training
# ==========================================
print("Training the Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# STEP 6: Regression Evaluation
# ==========================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# ==========================================
# STEP 7: Categorical Evaluation (Confusion Matrix)
# ==========================================
print("\n--- Categorical Evaluation ---")
bins = [0, 4, 7, 10]
category_labels = ['Poor (0-4)', 'Average (4-7)', 'Good (7-10)']

y_test_classes = pd.cut(y_test, bins=bins, labels=category_labels, include_lowest=True)
y_pred_classes = pd.cut(y_pred, bins=bins, labels=category_labels, include_lowest=True)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Categorical Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test_classes, y_pred_classes, labels=category_labels)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_labels)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title("Confusion Matrix: Movie Rating Categories")
plt.show()

# ==========================================
# STEP 8: Feature Importance Insights
# ==========================================
# IMPROVEMENT 3: Visualize which features the model relied on the most
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] # Sort in descending order

plt.figure(figsize=(10, 6))
plt.title("Feature Importances: What drives a movie's rating?")
plt.bar(range(X.shape[1]), importances[indices], color='teal', align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel("Importance Score")
plt.tight_layout() # Ensures labels don't get cut off
plt.show()