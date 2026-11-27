import pandas as pd
import joblib
from flask import Flask, render_template, request

# Load Model & Files

model = joblib.load("store_sales_xgboost_model.pkl")
preprocessor = joblib.load("preprocessor_column_transfer.pkl")

df_test = pd.read_csv("df_test_merged.csv")

# Flask App

app = Flask(__name__)

# Home Page

@app.route("/")
def home():
    return render_template("index_form.html")

# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Row Number
        row_no = int(request.form["row_no"])

        # Check valid range
        if row_no < 0 or row_no >= len(df_test):

            return render_template(
                "index_form.html",
                error=f"Please enter a row number between 0 and {len(df_test)-1}"
            )

        # Get Selected Row
        row = df_test.iloc[row_no]

        # Convert to DataFrame
        input_data = pd.DataFrame([row])

        # Preprocessing
        transformed_data = preprocessor.transform(input_data)

        # Prediction
        prediction = model.predict(transformed_data)[0]

        # Remove negative prediction
        if prediction < 0:
            prediction = 0

        prediction = round(prediction, 2)

        # Send to HTML
        return render_template(
            "index_form.html",
            row=row,
            row_no=row_no,
            prediction=prediction
        )

    except Exception as e:

        return render_template(
            "index_form.html",
            error=str(e)
        )

# Run App

if __name__ == "__main__":
    app.run(debug=True)