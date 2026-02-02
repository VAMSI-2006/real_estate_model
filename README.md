🏡 Real Estate Price Prediction Web Application
📌 Project Overview

This project is an end-to-end machine learning web application that predicts real estate prices in Bengaluru based on property features such as location, total square feet, number of bedrooms (BHK), and bathrooms.

The system compares multiple machine learning models and deploys the best-performing model (Random Forest) through an interactive Streamlit web interface for real-time price estimation.

🎯 Key Objectives

Predict house prices accurately using historical real estate data

Compare multiple ML models and identify the optimal one

Build an interactive, user-friendly web application for predictions

Demonstrate end-to-end ML workflow: data → model → deployment

🧠 Machine Learning Models Used

Linear Regression

Decision Tree Regressor

Random Forest Regressor ✅ (Best Performing)

📊 Model Performance Summary

Trained and evaluated models on 13,300+ Bengaluru housing records

Random Forest achieved ~89% R² score

Reduced prediction error by ~14% compared to baseline Linear Regression

Evaluation metrics used:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

🛠️ Data Preprocessing & Feature Engineering

Cleaned and normalized mixed square-footage units

Extracted numerical features such as BHK

Removed ~8% noisy/outlier records to improve data quality

Applied One-Hot Encoding to high-cardinality location data

Generated 40+ engineered features for improved prediction accuracy

🌐 Web Application (Streamlit)

The project includes an interactive Streamlit-based web application that allows users to:

Enter property details (location, sqft, BHK, bathrooms)

Select machine learning models

View real-time predicted prices

Compare model-driven insights visually

🧰 Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib

Web Framework: Streamlit

Model Serialization: Joblib

Tools: VS Code, Virtual Environment (venv)
