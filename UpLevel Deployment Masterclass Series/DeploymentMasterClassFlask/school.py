from flask import Flask, request
import json
import pandas as pd
import joblib

app = Flask(__name__)
forest = joblib.load("forest_v1.joblib")

# homepage
@app.route("/")
def index(): # what happens when you arrive at this route
    return "Hello world!"

# API endpoint
@app.route("/predict", methods=['GET']) # can deliver payload to API endpoint
def predict():
    # take API payload as JSON format
    json_ = request.json
    # JSON to df
    df = pd.read_json(json_)
    prediction = forest.predict(df)
    # turn prediction np array into a proper list
    # pack list into a dictionary
    return {"prediction": list(prediction)}

# app.run()
# go to
# http://127.0.0.1:5000/

if __name__ == '__main__':
    app.run()