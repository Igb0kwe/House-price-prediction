from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import joblib as jb
import pandas as pd
import os

# Load the saved models
modelmaid = jb.load("maidprice_model.pkl")
modelabj = jb.load("abjprice_model.pkl")
modellag = jb.load("lagprice_model.pkl")
modelkano = jb.load("kanoprice_model.pkl")

# Create Flask app
app = Flask(__name__)

# Define home route to display an input form
@app.route("/")
def home():
    return render_template("index.html")  # HTML file for user input

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        area_input = request.form["area"]
        location_input = request.form["location"]

        # Process area input
        area = float(area_input)
        area = np.array([[area]])

        # Handle prediction based on location
        if location_input == "Abuja":
            prediction = modelabj.predict(area)
        elif location_input == "Lagos":
            prediction = modellag.predict(area)
        elif location_input == "Maiduguri":
            prediction = modelmaid.predict(area)
        elif location_input == "Kano":
            prediction = modelkano.predict(area)
        else:
            raise ValueError("Invalid location selected.")
            
        #Process output data
        predicted_price = int(round(prediction[0], 0))
        predicted_price = f"{predicted_price:,}"
        
        area = float(area[0][0])
        
        # Render the output page with the results
        return render_template("output.html", area=area, predicted_price=predicted_price, location_input = location_input)

    except Exception as e:
        # Render the error page for any exception
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    # Bind to host 0.0.0.0 and use the PORT environment variable for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)