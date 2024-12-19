# House Price Prediction Web Application

This project is a machine learning-based web application that predicts house prices based on user inputs such as the number of bedrooms, bathrooms, and the living area in square feet. The application is built using Flask for the backend and HTML/CSS for the frontend.

# Features

Model Training: The house price prediction model is trained using linear regression on a dataset.

Web Interface: A user-friendly web application interface to input house details and get price predictions.

# Technologies Used

# Backend:

Python

Flask

Scikit-learn (for training the machine learning model)

# Frontend:

HTML/CSS

# Additional Libraries:

Pandas (for data manipulation)

Pickle (for saving and loading the trained model)

# File Structure

House_Price_Prediction/
|
|-- templates/

|   |-- index.html         # Frontend HTML file
|

|-- app.py                 # Flask application file
|

|-- model.pkl              # Trained machine learning model
|

|-- requirements.txt       # List of dependencies
|

|-- README.md              # Project documentation

# Usage

Open the application in your browser.

Enter the number of bedrooms, bathrooms, and the living area in square feet.

Click the "Predict Price" button to get the estimated house price in INR.
