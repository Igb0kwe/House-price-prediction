from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import joblib as jb
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os  # For accessing environment variables

# Load the saved model
model = jb.load("price_model.pkl")

df = pd.read_csv("homeprices.csv")

# Create Flask app
app = Flask(__name__)

# Define home route to display an input form
@app.route("/")
def home():
    return render_template("index.html")  # HTML file for user input

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    global df
    try:
        # Extract the area from the form input
        area_input = request.form["area"]

        # Validate input type
        try:
            area = float(area_input)  # Convert input to float
        except ValueError:
            raise ValueError("Input must be a numeric value.")

        # Prepare the data in a 2D array and make the prediction
        features_array = np.array([[area]])
        prediction = model.predict(features_array)
        predicted_price = round(prediction[0], 2)

        new_data = pd.DataFrame({"Area": [area], "Price": [predicted_price]})
        if not ((df["Area"] == area) & (df["Price"] == predicted_price)).any():
            df = pd.concat([df, new_data], ignore_index=True)
            
        highlight_area = area
        highlight_price = df[df['Area'] == highlight_area]['Price'].values[0]
        plt.scatter(highlight_area, highlight_price, color='blue', s=100, label= f"({area}sq ft, ${predicted_price})")
        plt.xlabel('Area(sqr ft)')
        plt.ylabel('Price(US$)')
        plt.scatter(df.Area, df.Price, color = 'black', marker='+')
        plt.title('Scatter Plot with Highlighted Point')
        plt.legend()

        # Render the output page with the results
        return render_template("output.html", area=area, predicted_price = predicted_price)
    
    except Exception as e:
        # Render the error page for any exception
        return render_template("error.html", error_message=str(e))
    

if __name__ == "__main__":
    # Bind to host 0.0.0.0 and use the PORT environment variable for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
