import flask as f
from pickle import load
from flask import url_for
import numpy as np
import os

# # Create a web application and load contents of pickle file
app = f.Flask(__name__)
model = load(open("bostMdl.pkl", "rb"))


# Load template on opening app home page
@app.route("/")
def home():
    return f.render_template("index.html")


# Create predictions and show it on page
@app.route("/predict", methods=["POST"])
def predict():
    features = []
    for i in f.request.form.values():
        features.append(float(i))
    # [[features[0],features[1], features[2], features[3], features[4], features[5]]]
    # predict_price = round(model.predict([features])[0], 2)
    # predict_price = round(model.predict(features)[0], 2)
    print(features)

    predict_price = round(
        model.predict([[features[0], features[1], features[2], features[3], features[4], features[5]]])[0], 2)

    return f.render_template("result.html", pred = predict_price)


if __name__ == "__main__":
    app.run(debug=True)
