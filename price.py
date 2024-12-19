import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle # for saving and loading the model

# Load dataset
df = pd.read_csv('data.csv')  # Replace with the dataset path
print(df.head())

# Check for missing values
print("Columns in the dataset:", df.columns)
print("Missing values in each column:\n", df.isnull().sum())

# Fill missing values (for numeric columns only)
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

# Encode categorical data (example: city and statezip)
df = pd.get_dummies(df, drop_first=True)

# Select features and target
# Updated feature columns based on your dataset
X = df[['sqft_lot', 'condition', 'yr_built']]  # Replace with other columns if necessary
# Update target column (ensure it exists in your dataset)
if 'price' in df.columns:
    y = df['price']
else:
    raise ValueError("Target column 'price' is missing from the dataset.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'house_price_model.pkl'")

with open('house_price_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)



# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Compare predictions with actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

