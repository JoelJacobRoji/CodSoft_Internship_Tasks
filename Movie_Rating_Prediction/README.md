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
