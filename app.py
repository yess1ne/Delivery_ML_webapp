from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load artifacts: model, scaler, and feature list
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_regressor_model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
features = artifact["features"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure all required fields are present
        missing = [f for f in features if f not in request.form]
        if missing:
            return f"Missing fields: {', '.join(missing)}", 400

        # Convert inputs safely
        try:
            form_data = {f: float(request.form[f]) for f in features}
        except ValueError as ve:
            return f"Invalid input. Please enter numeric values only. Details: {str(ve)}", 400

        # Prepare input DataFrame in correct order
        X_input = pd.DataFrame([form_data])[features]

        # Scale
        X_scaled = scaler.transform(X_input)

        # Predict
        prediction = round(model.predict(X_scaled)[0], 2)

        return render_template("result.html", prediction=prediction, inputs=form_data)

    except Exception as e:
        return f"Unexpected error: {str(e)}", 500


@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
