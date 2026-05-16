# 🎬 IMDb Movie Rating Prediction (Regression Analysis)

## 📌 Project Overview
This project is a **Supervised Regression** task. The objective is to build a predictive Machine Learning model that analyzes the characteristics of an Indian movie — such as its genre, director, actors, duration, and votes — to accurately estimate the IMDb rating (from **1 to 10**) it is likely to receive.

Unlike **Classification**, which measures success using exact categorical matches (**pass/fail**), **Regression** measures success based on **distance** — how close the predicted numerical value is to the actual rating.

Therefore, the model is evaluated using **error-based metrics** instead of percentage accuracy alone.

---

# 📊 The Dataset & Challenges

The raw IMDb India dataset is highly unstructured and required extensive preprocessing before machine learning could be applied.

## ⚠️ Major Challenges Faced

### 🧹 Dirty Text Data
Several numerical columns contained unnecessary text formatting:

| Column | Raw Example |
|---|---|
| Year | `(2019)` |
| Duration | `150 min` |
| Votes | `1,500` |

These values could not be processed directly by ML algorithms.

---

### ❌ Missing Data
Many movies did not contain valid IMDb ratings.

Since supervised learning requires a correct answer key (**ground truth**), rows with missing ratings had to be removed before training.

---

# ⚙️ Methodology & Blueprint

## 1️⃣ Data Cleaning & RegEx Processing

To clean the dataset, **Regular Expressions (RegEx)** were heavily used.

### ✅ Technique Used

```python
.str.extract('(\d+)')
```

This extracted only numeric values from messy text columns.

---

## 🔍 Example Transformations

| Raw Value | Cleaned Value |
|---|---|
| `(2019)` | `2019` |
| `150 min` | `150` |
| `1,500` | `1500` |

After extraction:
- Columns were converted into numeric datatypes
- Invalid rows were removed
- Missing target values were dropped

---

# 🏗️ 2️⃣ Feature Engineering (Target Encoding)

Machine Learning models cannot understand raw text values such as:
- Director Names
- Actor Names
- Genre Labels

To solve this, **Target Encoding** was implemented.

---

## 🔢 What is Target Encoding?

Each categorical value was replaced with its historical average IMDb rating.

### 📌 Example

```python
"Director Name" → 8.5
```

If a director's previous movies averaged **8.5**, then the algorithm mathematically represents that director using the value **8.5**.

This transforms human reputation into meaningful numerical intelligence.

---

# 🤖 3️⃣ Model Building & Evaluation

## ⚙️ Algorithm Used
- Random Forest Regressor

The Random Forest objectively determines:
- Which variables matter most
- Which features influence ratings
- Which relationships are non-linear

The model discovers mathematical patterns between:
- Directors
- Actors
- Genres
- Votes
- Runtime
- Release Year

and the final IMDb rating.

---

# 📏 Evaluation Metrics

Since this is a Regression problem, the model was evaluated using numerical error metrics.

---

## 📉 Mean Absolute Error (MAE)
Measures the average prediction error distance.

➡️ Lower MAE = Better predictions.

---

## 📉 Root Mean Squared Error (RMSE)
Penalizes larger mistakes more aggressively.

➡️ Useful for identifying unstable predictions.

---

## 📈 R² Score (Coefficient of Determination)
Measures how well the model explains the dataset variance.

### Interpretation

| Score | Meaning |
|---|---|
| 1.0 | Perfect Fit |
| 0.0 | No Predictive Power |

---

# 🚀 Key Enhancements (The "Wow" Factor)

To elevate the project beyond a basic MVP, several advanced improvements were implemented.

---

# 1️⃣ Noise Reduction (Data Filtering)

Movies with fewer than **50 votes** were filtered out.

---

## 🔍 Why Was This Necessary?

A movie with only 5 votes from friends or family members could artificially show a perfect **10/10 rating**, introducing misleading noise into the dataset.

### ✅ Result
Although the raw metric scores shifted slightly, the model became:
- More stable
- More realistic
- More trustworthy for predicting widely-watched films

This improved the model's practical business reliability.

---

# 2️⃣ Regression to Classification (Binning)

A standard **Confusion Matrix** cannot directly evaluate continuous Regression outputs.

To solve this, predicted ratings were converted into business-friendly text categories.

---

## 🎭 Rating Categories

| Rating Range | Category |
|---|---|
| 0 – 4 | Poor |
| 4 – 7 | Average |
| 7 – 10 | Good |

---

## ✅ Benefit
This enabled the creation of a **Categorical Confusion Matrix**, helping stakeholders visualize prediction quality using understandable business terminology instead of raw decimal values.

---

# 3️⃣ Advanced EDA & Feature Importance

Several visual analytics were added to improve explainability and presentation quality.

---

## 📊 Visualizations Included
- IMDb Rating Distribution Graph
- Correlation Analysis
- Feature Importance Bar Charts

---

## 🔍 Feature Importance Insights

The Random Forest mathematically ranked:
- Directors
- Actors
- Genres
- Votes
- Duration
- Release Year

based on their influence over final IMDb ratings.

This improves model transparency and helps stakeholders understand *why* predictions are made.

---

# 💻 The UI Addition & Future Deployment

An interactive **Tkinter GUI** was developed as a proof-of-concept application.

---

# 🖥️ Advanced GUI Features

## 🔄 1. Fallback System

If the user enters:
- An unknown director
- An unseen actor
- A missing category

the application automatically assigns the **Global Average Rating** to maintain stable mathematical predictions.

### ✅ Why This Matters
Without fallback handling:
- The model could crash
- Predictions could become invalid
- Unseen categories could break the pipeline

This significantly improves real-world usability.

---

## 🎭 2. Genre Splitting System

Instead of requiring exact genre matches such as:

```python
"Action, Sci-Fi"
```

the genre input was split into multiple independent feature columns.

### ✅ Benefit
This allows the model to:
- Understand mixed genres individually
- Evaluate the influence of each genre separately
- Handle partial genre combinations more effectively

For example:
- Action
- Drama
- Sci-Fi

can each contribute independently to the prediction.

---

## 📈 3. Outlier Handling Context

The project also addressed scenarios where user inputs fall far outside the training dataset's "comfort zone."

### 📌 Example

```python
120,000 votes for a 2024 movie
```

when the training data ends in **2021**.

---

## ✅ Model Behavior
In these situations, the Random Forest relies more heavily on:
- Encoded director reputation
- Genre averages
- Actor historical performance

instead of unstable numerical assumptions.

This improves robustness during unpredictable inputs.

---

# 🚀 Future Scope

The GUI version serves as a strong deployment-ready prototype.

Future improvements may include:
- Live IMDb API integration
- Real-time industry trend updates
- Automatic dataset refreshing
- Continuous model retraining
- Cloud deployment for public access

---

# 🛠️ Technologies & Libraries Used

## 💻 Programming Language
- Python 3.x

---

## 📦 Data Manipulation
- pandas
- numpy

---

## 📊 Data Visualization
- matplotlib
- seaborn

---

## 🤖 Machine Learning
- scikit-learn
  - Random Forest Regressor
  - Train-Test Split
  - Evaluation Metrics
  - Feature Engineering

---

## 🖥️ GUI Development
- tkinter

---

## 🔍 Data Cleaning
- Regular Expressions (RegEx)

---

# 📂 Project Structure

```bash
IMDb_Movie_Rating_Prediction/
│
├── dataset/
│   └── IMDb_India_Movies.csv
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── visuals/
│   ├── rating_distribution.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── gui/
│   └── movie_rating_gui.py
│
├── model/
│   └── random_forest_model.pkl
│
├── requirements.txt
└── README.md
```

---

# ▶️ How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/imdb-movie-rating-prediction.git
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run the Notebook

```bash
jupyter notebook
```

Open:
```bash
EDA_and_Modeling.ipynb
```

---

## 4️⃣ Launch the GUI

```bash
python movie_rating_gui.py
```

---

# ⭐ Final Outcome

This project successfully demonstrates:

- Advanced Data Cleaning
- Feature Engineering
- Supervised Regression
- Explainable AI Concepts
- Error-Based Evaluation Metrics
- Interactive ML Deployment
- Robust Handling of Real-World Noisy Data

More importantly, it highlights the ability to transform messy entertainment-industry data into a reliable predictive intelligence system capable of generating meaningful business insights and production-ready decision support.
