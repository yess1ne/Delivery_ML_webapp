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
df_clustering = pd.read_csv("datadelevry_cleaned.csv")
clustering_model= joblib.load("clustering.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/clustering_index", methods=["GET"])
def clustering_index():
    stats = df_clustering.describe().to_html(classes='table table-striped', border=0)
    return render_template('clustering_index.html', tables=stats)

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

@app.route('/predict_clustering', methods=['GET', 'POST'])
def predict_clustering():
    if request.method == 'POST':
        # Récupérer les valeurs du formulaire
        try:
            input_data = {
    'total_orders': float(request.form['total_orders']),
    'avg_order_value': float(request.form['avg_order_value']),
    'avg_items_per_order': float(request.form['avg_items_per_order']),
    'avg_delivery_time': float(request.form['avg_delivery_time']),
    'driver_utilization_rate': float(request.form['driver_utilization_rate']),
    'price_range': float(request.form['price_range']),
    'revenue_per_item': float(request.form['revenue_per_item']),
    'delivery_time_consistency': float(request.form['delivery_time_consistency']),
    'operational_efficiency': float(request.form['operational_efficiency'])
}

            # Transformer en DataFrame
            X_new = pd.DataFrame([input_data])

            # Prédire le cluster
            cluster = clustering_model.predict(X_new)[0]
            return render_template('clustering_result.html', cluster=cluster, data=input_data)

        except Exception as e:
            return f"Erreur lors de la prédiction : {e}"

    return render_template('clustering_predict.html')

@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
