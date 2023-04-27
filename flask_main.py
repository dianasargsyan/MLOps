from flask import Flask, request, jsonify
import pandas as pd
import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.utils import check_missing_values, clean_car_names, convert_symboling_to_string, select_features

checkpoint_file = './checkpoints/model_checkpoint2.joblib'
model = joblib.load(checkpoint_file)


def home():
    return "Use this API to predict car prices."


def predict_price(wheelbase, carlength, carwidth, carheight, curbweight, enginesize, 
                  boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg):

    # Load the feature ranges
    input_df = pd.DataFrame({
        'wheelbase': [wheelbase],
        'carlength': [carlength],
        'carwidth': [carwidth],
        'carheight': [carheight],
        'curbweight': [curbweight],
        'enginesize': [enginesize],
        'boreratio': [boreratio],
        'stroke': [stroke],
        'compressionratio': [compressionratio],
        'horsepower': [horsepower],
        'peakrpm': [peakrpm],
        'citympg': [citympg],
        'highwaympg': [highwaympg]
    })
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_df)
    return jsonify({'predicted_price': prediction[0]})
