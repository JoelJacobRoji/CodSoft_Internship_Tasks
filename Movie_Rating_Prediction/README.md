🎬 IMDb Movie Rating Prediction (Regression Analysis)

📌 Project Overview
This project is a Supervised Regression task. The objective is to build a predictive machine-learning model that analyzes the characteristics of an Indian movie (such as its genre, director, actors, duration, and votes) to accurately estimate the rating (from 1 to 10) it will receive on IMDb.

While Classification measures success by exact categorical matches (pass/fail), Regression measures success by distance (how close the prediction is to the actual continuous numerical value). Therefore, the model's accuracy is evaluated using error rates rather than simple percentage matching.

📊 The Dataset & Challenges
The raw IMDb India dataset is notoriously messy and requires heavy data preprocessing before any machine learning can occur:

Dirty Text: Years enclosed in parentheses (2019), durations containing text "150 min", and votes containing commas "1,500".

Missing Data: Many movies lacked actual ratings, requiring initial filtering to ensure the model only trained on valid ground-truth data.

⚙️ Methodology & Blueprint
1. Data Cleaning & Regex
Utilized Regular Expressions (RegEx), specifically .str.extract('(\d+)'), to strip away non-numeric characters from the Year, Duration, and Votes columns, converting them into pure usable floats.

Dropped rows where the target variable (Rating) was completely missing.

2. Feature Engineering (Target Encoding)
Machine learning models only understand numbers, not text. To process the Director, Genre, and Actor columns:

Target Encoding: Replaced categorical text strings with the historical average rating associated with that category. (e.g., If a specific director's past movies average an 8.5, their name is mathematically represented as 8.5 in the algorithm).

3. Model Building & Evaluation
Algorithm: Trained a Random Forest Regressor. The model objectively finds the mathematical truth by assessing which features (Director vs. Genre vs. Votes) actually drive a movie's success.

Metrics: Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R2 Score (where 1.0 represents a perfect fit).

🚀 Key Enhancements (The "Wow" Factor)
1. Noise Reduction (Data Filtering)
Filtered out movies with fewer than 50 votes. Previously, a movie with 5 votes from the director's family might have an artificial 10/10 rating. Dropping these removed "fake" ratings. While the overall raw metric scores shifted, the model became significantly more robust and honest at predicting the ratings of real, widely-watched movies.

2. Regression to Classification (Binning)
Because you cannot directly use a Confusion Matrix for a continuous Regression model, the numerical predictions were binned into text categories:

0 to 4 = Poor

4 to 7 = Average

7 to 10 = Good 

This allowed for the generation of a Categorical Confusion Matrix to visualize the model's practical accuracy in standard business terms.

3. Advanced EDA & Feature Importance
Included visualizations showing the overall distribution of IMDb ratings.

Generated a Feature Importance Chart to visually rank to stakeholders exactly which variables the Random Forest relied on the most to make its predictions.

💻 The UI Addition & Future Deployment
An interactive GUI (using Tkinter) was developed as a proof-of-concept with several advanced features:

Fallback System: If a user inputs an unknown Director or Actor not present in the dataset, the UI automatically assigns the "Global Average Rating" to keep the math stable.

Genre Splitting: Instead of requiring exact string matches (e.g., "Action, Sci-Fi"), the genre input was split into three separate feature columns. This partial matching allows the model to simultaneously evaluate the impact of multiple distinct genres.

Outlier Handling Context: Addressed the reality that inputs wildly outside the training data's "comfort zone" (e.g., 120,000 votes for a 2024 movie when the dataset ends in 2021) require the model to rely heavier on text category averages.

Note: The MVP and Enhanced EDA version serve as the primary demonstration for this project. The UI version remains a strong proof-of-concept ready for future deployment when linked to a live, automatically updating dataset of new directors and changing industry factors.