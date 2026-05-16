# Considers genre as one feature, which is not ideal. A movie with "Action, Comedy" is very different from "Action, Horror".

'''


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk, messagebox

# ==========================================
# PART 1: DATA PREPARATION & MODEL TRAINING
# ==========================================
print("\nLoading data and training model... Please wait a moment.")

# 1. Load Data
try:
    df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
except FileNotFoundError:
    print("ERROR: 'IMDb Movies India.csv' not found in the current folder.")
    exit()

# 2. Clean Data
df = df.dropna(subset=['Rating'])
df['Year'] = df['Year'].str.extract(r'(\d+)').astype(float)
df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)
df['Votes'] = df['Votes'].str.replace(',', '').astype(float)

df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Noise Reduction
df = df[df['Votes'] >= 50]

# 3. Feature Engineering (Creating Dictionaries for the UI)
# We need to save the averages so the UI can look them up later!
global_mean_rating = df['Rating'].mean()

categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
encoding_dicts = {} # This will store the averages for every director, actor, etc.

for feature in categorical_features:
    df[feature] = df[feature].fillna('Unknown')
    # Calculate means and convert to a dictionary
    mean_encodings = df.groupby(feature)['Rating'].mean().to_dict()
    encoding_dicts[feature] = mean_encodings
    # Apply to dataframe for training
    df[feature + '_encoded'] = df[feature].map(mean_encodings)

# 4. Train the Model
features = ['Year', 'Duration', 'Votes', 'Genre_encoded', 'Director_encoded', 
            'Actor 1_encoded', 'Actor 2_encoded', 'Actor 3_encoded']

X = df[features]
y = df['Rating']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y) # Training on all available filtered data for maximum UI accuracy

print("Model trained successfully! Launching UI...")

# ==========================================
# PART 2: THE GRAPHICAL USER INTERFACE (GUI)
# ==========================================

def predict_rating():
    """This function runs when the user clicks the 'Predict' button."""
    try:
        # 1. Get values from the UI text boxes
        year = float(entry_year.get())
        duration = float(entry_duration.get())
        votes = float(entry_votes.get())
        genre = entry_genre.get()
        director = entry_director.get()
        actor1 = entry_actor1.get()
        actor2 = entry_actor2.get()
        actor3 = entry_actor3.get()
        
        # 2. Encode the text inputs using our saved dictionaries.
        # If the text is NOT in the dictionary (a new director/actor), use the global_mean_rating.
        gen_enc = encoding_dicts['Genre'].get(genre, global_mean_rating)
        dir_enc = encoding_dicts['Director'].get(director, global_mean_rating)
        act1_enc = encoding_dicts['Actor 1'].get(actor1, global_mean_rating)
        act2_enc = encoding_dicts['Actor 2'].get(actor2, global_mean_rating)
        act3_enc = encoding_dicts['Actor 3'].get(actor3, global_mean_rating)
        
        # 3. Format the data for the model
        input_data = pd.DataFrame([[year, duration, votes, gen_enc, dir_enc, act1_enc, act2_enc, act3_enc]], 
                                  columns=features)
        
        # 4. Make the prediction
        prediction = model.predict(input_data)[0]
        
        # 5. Update the UI with the result
        result_label.config(text=f"Predicted IMDb Rating: {prediction:.1f} / 10", fg="#2e7d32")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please make sure Year, Duration, and Votes are numbers!")

# --- Build the Main Window ---
root = tk.Tk()
root.title("IMDb Movie Rating Predictor")
root.geometry("450x600")
root.configure(padx=20, pady=20, bg="#f5f5f5")

# Title Label
title_label = tk.Label(root, text="🎥 Movie Rating Predictor", font=("Helvetica", 18, "bold"), bg="#f5f5f5", fg="#333")
title_label.pack(pady=(0, 20))

# Create a frame for the input fields
frame = tk.Frame(root, bg="#f5f5f5")
frame.pack(fill="both", expand=True)

# Helper function to create rows of labels and inputs
def create_input_row(parent, label_text, default_val=""):
    row = tk.Frame(parent, bg="#f5f5f5")
    row.pack(fill="x", pady=5)
    lbl = tk.Label(row, text=label_text, width=15, anchor="w", bg="#f5f5f5", font=("Helvetica", 10))
    lbl.pack(side="left")
    entry = ttk.Entry(row, font=("Helvetica", 10))
    entry.insert(0, default_val)
    entry.pack(side="right", expand=True, fill="x")
    return entry

# Create Inputs (Filled with a default example so you can test it immediately)
entry_title = create_input_row(frame, "Movie Title:", "Dangal") # Title is just for user experience!
entry_year = create_input_row(frame, "Release Year:", "2016")
entry_duration = create_input_row(frame, "Duration (mins):", "161")
entry_votes = create_input_row(frame, "Expected Votes:", "150000")
entry_genre = create_input_row(frame, "Genre:", "Action, Biography, Drama")
entry_director = create_input_row(frame, "Director:", "Nitesh Tiwari")
entry_actor1 = create_input_row(frame, "Actor 1:", "Aamir Khan")
entry_actor2 = create_input_row(frame, "Actor 2:", "Sakshi Tanwar")
entry_actor3 = create_input_row(frame, "Actor 3:", "Fatima Sana Shaikh")

# Predict Button
predict_btn = tk.Button(root, text="Predict Rating", font=("Helvetica", 12, "bold"), bg="#1976d2", fg="white", 
                        command=predict_rating, relief="flat", padx=10, pady=5)
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="Predicted IMDb Rating: --", font=("Helvetica", 16, "bold"), bg="#f5f5f5", fg="#555")
result_label.pack(pady=10)

# Start the application
root.mainloop()


'''

# Considers genre as seperate features, which is better than treating it as one. 


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import ttk, messagebox

# ==========================================
# PART 1: DATA PREPARATION & MODEL TRAINING
# ==========================================
print("\nLoading data and training model... Please wait a moment.")

# 1. Load Data
try:
    df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
except FileNotFoundError:
    print("ERROR: 'IMDb Movies India.csv' not found in the current folder.")
    exit()

# 2. Clean Data
df = df.dropna(subset=['Rating'])
df['Year'] = df['Year'].str.extract(r'(\d+)').astype(float)
df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)
df['Votes'] = df['Votes'].str.replace(',', '').astype(float)

df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Noise Reduction
df = df[df['Votes'] >= 50]

# --- NEW: Give Genre More Importance by Splitting It ---
df['Genre'] = df['Genre'].fillna('Unknown')
# Split the comma-separated string into up to 3 separate columns
genres_split = df['Genre'].str.split(', ', n=2, expand=True)
df['Genre_1'] = genres_split[0].fillna('Unknown')
df['Genre_2'] = genres_split[1].fillna('Unknown')
df['Genre_3'] = genres_split[2].fillna('Unknown')

# 3. Feature Engineering (Creating Dictionaries for the UI)
# We need to save the averages so the UI can look them up later!
global_mean_rating = df['Rating'].mean()

# Replace 'Genre' with the 3 new separated genre columns
categorical_features = ['Genre_1', 'Genre_2', 'Genre_3', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
encoding_dicts = {} # This will store the averages for every director, actor, etc.

for feature in categorical_features:
    df[feature] = df[feature].fillna('Unknown')
    # Calculate means and convert to a dictionary
    mean_encodings = df.groupby(feature)['Rating'].mean().to_dict()
    encoding_dicts[feature] = mean_encodings
    # Apply to dataframe for training
    df[feature + '_encoded'] = df[feature].map(mean_encodings)

# 4. Train the Model
# Update features list to include the 3 new encoded genre columns
features = ['Year', 'Duration', 'Votes', 'Genre_1_encoded', 'Genre_2_encoded', 'Genre_3_encoded', 'Director_encoded', 
            'Actor 1_encoded', 'Actor 2_encoded', 'Actor 3_encoded']

X = df[features]
y = df['Rating']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y) # Training on all available filtered data for maximum UI accuracy

print("Model trained successfully! Launching UI...")

# ==========================================
# PART 2: THE GRAPHICAL USER INTERFACE (GUI)
# ==========================================

def predict_rating():
    """This function runs when the user clicks the 'Predict' button."""
    try:
        # 1. Get values from the UI text boxes
        year = float(entry_year.get())
        duration = float(entry_duration.get())
        votes = float(entry_votes.get())
        genre_input = entry_genre.get()
        director = entry_director.get()
        actor1 = entry_actor1.get()
        actor2 = entry_actor2.get()
        actor3 = entry_actor3.get()
        
        # --- NEW: Process the UI input genre into 3 parts ---
        genres_split = genre_input.split(', ')
        genre_1 = genres_split[0] if len(genres_split) > 0 else 'Unknown'
        genre_2 = genres_split[1] if len(genres_split) > 1 else 'Unknown'
        genre_3 = genres_split[2] if len(genres_split) > 2 else 'Unknown'

        # 2. Encode the text inputs using our saved dictionaries.
        # If the text is NOT in the dictionary (a new director/actor), use the global_mean_rating.
        gen1_enc = encoding_dicts['Genre_1'].get(genre_1, global_mean_rating)
        gen2_enc = encoding_dicts['Genre_2'].get(genre_2, global_mean_rating)
        gen3_enc = encoding_dicts['Genre_3'].get(genre_3, global_mean_rating)
        
        dir_enc = encoding_dicts['Director'].get(director, global_mean_rating)
        act1_enc = encoding_dicts['Actor 1'].get(actor1, global_mean_rating)
        act2_enc = encoding_dicts['Actor 2'].get(actor2, global_mean_rating)
        act3_enc = encoding_dicts['Actor 3'].get(actor3, global_mean_rating)
        
        # 3. Format the data for the model
        input_data = pd.DataFrame([[year, duration, votes, gen1_enc, gen2_enc, gen3_enc, dir_enc, act1_enc, act2_enc, act3_enc]], 
                                  columns=features)
        
        # 4. Make the prediction
        prediction = model.predict(input_data)[0]
        
        # 5. Update the UI with the result
        result_label.config(text=f"Predicted IMDb Rating: {prediction:.1f} / 10", fg="#2e7d32")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please make sure Year, Duration, and Votes are numbers!")

# --- Build the Main Window ---
root = tk.Tk()
root.title("IMDb Movie Rating Predictor")
root.geometry("450x600")
root.configure(padx=20, pady=20, bg="#f5f5f5")

# Title Label
title_label = tk.Label(root, text="🎥 Movie Rating Predictor", font=("Helvetica", 18, "bold"), bg="#f5f5f5", fg="#333")
title_label.pack(pady=(0, 20))

# Create a frame for the input fields
frame = tk.Frame(root, bg="#f5f5f5")
frame.pack(fill="both", expand=True)

# Helper function to create rows of labels and inputs
def create_input_row(parent, label_text, default_val=""):
    row = tk.Frame(parent, bg="#f5f5f5")
    row.pack(fill="x", pady=5)
    lbl = tk.Label(row, text=label_text, width=15, anchor="w", bg="#f5f5f5", font=("Helvetica", 10))
    lbl.pack(side="left")
    entry = ttk.Entry(row, font=("Helvetica", 10))
    entry.insert(0, default_val)
    entry.pack(side="right", expand=True, fill="x")
    return entry

# Create Inputs (Filled with a default example so you can test it immediately)
entry_title = create_input_row(frame, "Movie Title:", "Dangal") # Title is just for user experience!
entry_year = create_input_row(frame, "Release Year:", "2016")
entry_duration = create_input_row(frame, "Duration (mins):", "161")
entry_votes = create_input_row(frame, "Expected Votes:", "150000")
entry_genre = create_input_row(frame, "Genre:", "Action, Biography, Drama")
entry_director = create_input_row(frame, "Director:", "Nitesh Tiwari")
entry_actor1 = create_input_row(frame, "Actor 1:", "Aamir Khan")
entry_actor2 = create_input_row(frame, "Actor 2:", "Sakshi Tanwar")
entry_actor3 = create_input_row(frame, "Actor 3:", "Fatima Sana Shaikh")

# Predict Button
predict_btn = tk.Button(root, text="Predict Rating", font=("Helvetica", 12, "bold"), bg="#1976d2", fg="white", 
                        command=predict_rating, relief="flat", padx=10, pady=5)
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="Predicted IMDb Rating: --", font=("Helvetica", 16, "bold"), bg="#f5f5f5", fg="#555")
result_label.pack(pady=10)

# Start the application
root.mainloop()

