📈 Advertising Sales Prediction (Business Analytics)

📌 Project Overview
This project represents a transition from standard machine learning to Business-Focused Regression. The objective is to build a predictive model that acts as a financial tool: analyzing how much money is spent on TV, Radio, and Newspaper advertising to forecast the total units of product sold.

More than just predicting a number, the primary goal of this project is Budget Optimization—using data to advise stakeholders on exactly where their marketing dollars will yield the highest Return on Investment (ROI).

🧮 Model Selection: The "Glass Box" Approach
Unlike the Movie Rating project which used a Random Forest (a "Black Box" model where the internal logic is too complex for humans to trace), this project utilizes Simple & Multiple Linear Regression (a "Glass Box" model).

Why Linear Regression? It calculates a strict, transparent mathematical formula (e.g., Sales=intercept+(W1×TV)+(W2×Radio)). This transparency allows us to isolate the exact "weight" or multiplier for every single marketing channel.

How it works: The algorithm looks at historical patterns and runs millions of micro-calculations to find the perfect "weights" to multiply against each category (similar to adjusting the ingredients in a cake recipe until it perfectly matches a historical taste).

⚙️ Methodology & Blueprint
1. Exploratory Data Analysis (EDA)
Heatmaps: Used to identify exact correlation percentages. Revealed that TV spending has a massive correlation to sales (~0.7 to 0.9), while Newspaper spending is negligible.

Scatter Plots: Visually confirmed relationships (TV forms a tight, upward diagonal line indicating a strong relationship; Newspaper forms a random cloud of dots indicating a weak relationship).

2. Preprocessing & Evaluation
Separated features (X: Budgets) from the target (y: Sales).

Evaluated the mathematical fit using Mean Absolute Error (MAE) and the R2 Score.

Plotted a "Line of Best Fit" to visually demonstrate how closely the model's predictions align with reality.

💼 Strategic Business Insights (The ROI Formula)
By analyzing the model's generated Coefficients (slopes), we extracted the following business insights:

The Efficiency Winner (Radio): The coefficient for Radio (~0.10) is double that of TV (~0.05). Dollar-for-dollar, Radio is twice as efficient. For every $1 spent on Radio, the company gets 0.10 extra sales.

The Volume Driver (TV): TV drives the vast majority of total sales simply due to the massive disparity in budget allocation (Average TV Budget: ~$150,000 vs. Average Radio Budget: ~$23,000).

The Dead Weight (Newspaper): The coefficient is so low (~0.004) that it is statistically insignificant.

⚠️ The Real-World Strategy & Diminishing Returns
While pure math suggests we should move half the TV budget to Radio, Linear Regression has a flaw: It assumes lines go up forever. In the real world, dumping $500,000 into Radio would lead to Market Saturation (audience annoyance and drastically dropping ROI).

Final Actionable Recommendation:

Kill Newspaper: Reallocate 100% of the wasted Newspaper budget immediately to Radio.

A/B Testing / Gradual Shift: Shift only 10% of the TV budget to Radio for the next quarter. Re-evaluate the model with new data. If the Radio coefficient stays high, shift another 10%. Stop shifting when the coefficient begins to drop (signaling maximum saturation).

🚀 Key Enhancements
Interactive Budget Simulator (GUI): Developed a Tkinter desktop application that allows marketing managers to manually input hypothetical budgets for TV, Radio, and Newspaper to instantly see the forecasted sales volume based on the model's underlying formula.

🏃‍♂️ How to Run the App
Ensure advertising.csv is in the same directory as the Python script.

Run python sales_prediction.py from your terminal.

Review the EDA graphs and the terminal output for the raw ROI metrics.

Use the pop-up Interactive Budget Simulator to test budget reallocation strategies.