import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# STEP 1: Load Data
# ==========================================
# The IMDb India dataset often requires 'latin1' encoding due to special characters
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

print("\n\nInitial Data Shape:", df.shape)

# ==========================================
# STEP 2: Data Preprocessing & Cleaning
# ==========================================
# 2.1 Drop rows where the target variable 'Rating' is missing
df = df.dropna(subset=['Rating'])

# 2.2 Clean the 'Year' column (e.g., "(2019)" -> 2019)
df['Year'] = df['Year'].astype(str).str.extract(r'(\d+)').astype(float)

# 2.3 Clean the 'Duration' column (e.g., "109 min" -> 109)
df['Duration'] = df['Duration'].astype(str).str.replace(' min', '').astype(float)

# 2.4 Clean the 'Votes' column (e.g., "1,034" -> 1034)
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').astype(float)

# 2.5 Handle missing values in numerical features by filling with the median
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# ==========================================
# STEP 3: Feature Engineering (Target Encoding)
# ==========================================
# We will encode categorical features by replacing them with the mean rating of that category.
# For example, replace a Director's name with the average rating of all their movies.

categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for feature in categorical_features:
    # Fill missing text values with 'Unknown'
    df[feature] = df[feature].fillna('Unknown')
    
    # Calculate the mean rating for each category in the feature
    mean_encodings = df.groupby(feature)['Rating'].mean()
    
    # Map the mean ratings back to the dataframe
    # We add a suffix '_encoded' to keep the original columns if needed for EDA
    df[feature + '_encoded'] = df[feature].map(mean_encodings)

# Define our final features (X) and our target (y)
features = ['Year', 'Duration', 'Votes', 'Genre_encoded', 'Director_encoded', 
            'Actor 1_encoded', 'Actor 2_encoded', 'Actor 3_encoded']

X = df[features]
y = df['Rating']

# ==========================================
# STEP 4: Model Training
# ==========================================
# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# We use 100 trees (n_estimators)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
print("Training the model... this might take a few seconds.")
model.fit(X_train, y_train)

# ==========================================
# STEP 5: Evaluation
# ==========================================
# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---\n")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Optional: Visualize Actual vs Predicted Ratings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted IMDb Ratings")
plt.show()


from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

print("\n--- Regression to Classification using Classification Adapter for Categorical Evaluation : ---\n")


# 1. Define our rating categories (bins)
bins = [0, 4, 7, 10]
category_labels = ['Poor (0-4)', 'Average (4-7)', 'Good (7-10)']

# 2. Convert the continuous Actual and Predicted numbers into these text categories
# We use pd.cut() to slice the numbers into the bins we defined above
y_test_classes = pd.cut(y_test, bins=bins, labels=category_labels, include_lowest=True)
y_pred_classes = pd.cut(y_pred, bins=bins, labels=category_labels, include_lowest=True)

# 3. Calculate Categorical Accuracy
# This checks how often the model guessed the right bucket (e.g., guessed "Good" when it actually was "Good")
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Categorical Accuracy: {accuracy * 100:.2f}%\n")

# 4. Create the Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes, labels=category_labels)

# 5. Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_labels)

# Plotting with a blue color map
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())

plt.title("Confusion Matrix: Movie Rating Categories")
plt.xlabel("Predicted Rating Category")
plt.ylabel("Actual Rating Category")
plt.show()