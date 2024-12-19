from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the USD to INR conversion rate (update this value as needed)
USD_TO_INR = 83.0  # Example exchange rate

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the input values from the form
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])

        # Prepare the data for the model
        input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living]], 
                                  columns=['bedrooms', 'bathrooms', 'sqft_living'])

        # Make a prediction
        prediction_usd = model.predict(input_data)[0]

        # Convert the prediction to INR
        prediction_inr = prediction_usd * USD_TO_INR

        # Return the result to the web app
        return render_template(
            'index.html',
            prediction_text=f'Predicted House Price: ₹{prediction_inr:,.2f}'
        )
    except Exception as e:
        # Handle errors gracefully
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

