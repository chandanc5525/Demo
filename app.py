from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
model = joblib.load("modelpipeline.joblib")

# Define the columns the model expects
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Collect inputs from form
            data = [[
                float(request.form[col]) for col in columns
            ]]

            df = pd.DataFrame(data, columns=columns)

            # Make prediction
            pred = model.predict(df)[0]
            prediction = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
