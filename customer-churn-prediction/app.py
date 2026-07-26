#import joblib

#model = joblib.load("customer_churn_model.pkl")

#print("Model loaded successfully!")

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Loading the Model
gbc_Model = joblib.load("customer_churn_model.pkl")

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    Age = float(request.form["Age"])
    Login_Frequency = float(request.form["Login_Frequency"])
    Session_Duration_Avg = float(request.form["Session_Duration_Avg"])
    Pages_Per_Session = float(request.form["Pages_Per_Session"])
    Cart_Abandonment_Rate = float(request.form["Cart_Abandonment_Rate"])
    Wishlist_Items = float(request.form["Wishlist_Items"])
    Total_Purchases = float(request.form["Total_Purchases"])
    Average_Order_Value = float(request.form["Average_Order_Value"])
    Days_Since_Last_Purchase = float(request.form["Days_Since_Last_Purchase"])
    Discount_Usage_Rate = float(request.form["Discount_Usage_Rate"])
    Returns_Rate = float(request.form["Returns_Rate"])
    Email_Open_Rate = float(request.form["Email_Open_Rate"])
    Customer_Service_Calls = float(request.form["Customer_Service_Calls"])
    Product_Reviews_Written = float(request.form["Product_Reviews_Written"])
    Social_Media_Engagement_Score = float(request.form["Social_Media_Engagement_Score"])
    Mobile_App_Usage = float(request.form["Mobile_App_Usage"])
    Lifetime_Value = float(request.form["Lifetime_Value"])

    input_data = pd.DataFrame([{
        'Age': Age,
        'Login_Frequency': Login_Frequency,
        'Session_Duration_Avg': Session_Duration_Avg,
        'Pages_Per_Session': Pages_Per_Session,
        'Cart_Abandonment_Rate': Cart_Abandonment_Rate,
        'Wishlist_Items': Wishlist_Items,
        'Total_Purchases': Total_Purchases,
        'Average_Order_Value': Average_Order_Value,
        'Days_Since_Last_Purchase': Days_Since_Last_Purchase,
        'Discount_Usage_Rate': Discount_Usage_Rate,
        'Returns_Rate': Returns_Rate,
        'Email_Open_Rate': Email_Open_Rate,
        'Customer_Service_Calls': Customer_Service_Calls,
        'Product_Reviews_Written': Product_Reviews_Written,
        'Social_Media_Engagement_Score': Social_Media_Engagement_Score,
        'Mobile_App_Usage': Mobile_App_Usage,
        'Lifetime_Value': Lifetime_Value
    }])

    prediction = gbc_Model.predict(input_data)[0]
    probability = gbc_Model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "⚠️ Customer Likely to Churn"
    else:
        result = "✅ Customer Likely to Stay"

    probability = round(probability * 100, 2)

    return render_template("form.html",prediction_text=result,probability=probability)

if __name__ == "__main__":
    app.run(debug=True)