from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib as jb
import os  # For accessing environment variables

# Load the saved model
model = jb.load("price_model.pkl")

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
        # Extract the area from the form input
        area = float(request.form["area"])  # Expecting a single input for area
        features_array = np.array([[area]])  # Prepare the data in 2D array
        prediction = model.predict(features_array)

        return render_template("output.html", area=area, predicted_price=round(prediction[0], 2))
    except Exception as e:
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    # Bind to host 0.0.0.0 and use the PORT environment variable for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
