📌Medical Cost Prediction using Machine Learning

This project focuses on predicting medical insurance costs based on individual attributes such as age, BMI, smoking status, region, and number of children.

The objective is to apply Machine Learning techniques to:
Predict insurance charges (Regression).
Classify individuals into high-risk and low-risk categories (Classification).
Analyze feature importance influencing medical expenses.

📊 Dataset

The project uses the Insurance Dataset, which contains the following features:
age – Age of the individual.
sex – Gender.
bmi – Body Mass Index.
children – Number of dependents.
smoker – Smoking status.
region – Residential area.
charges – Medical insurance cost (Target Variable).

Exploratory Data Analysis (EDA)

Performed various visualizations to understand the dataset:
Distribution of insurance charges.
Smoker vs Charges comparison.
Age vs Charges relationship.
Correlation heatmap.
Feature importance analysis.
EDA helped identify strong relationships between smoking, age, BMI, and insurance costs.

🤖 Machine Learning Models Used

🔹 1. Random Forest Regressor
Used for predicting medical insurance charges.

Evaluated using:
R² Score
Mean Squared Error (MSE)

🔹 2. Decision Tree Regressor
Implemented for comparison with Random Forest.
Evaluated using R² Score and MSE.

🔹 3. Random Forest Classifier

Created a new feature risk:
High Risk (charges > 20000)
Low Risk (charges ≤ 20000)

Evaluated using:
Accuracy Score
Classification Report

Key Outcomes

Successfully built regression and classification models.
Identified smoking status as a major factor affecting medical costs.
Compared multiple models for performance evaluation.
Implemented feature importance visualization for model interpretability.

Technologies Used:

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Google Colab Link

You can run this project directly on Google Colab:
https://colab.research.google.com/drive/1-0kImtLHyxix-3aA5IYUYTRnw0-LU4Fq#scrollTo=CKsaVm0p5GtR
