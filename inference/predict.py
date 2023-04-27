import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.utils import check_missing_values, clean_car_names, convert_symboling_to_string, select_features
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib


pd.options.display.float_format = '{:.4f}'.format
plt.rcParams['figure.figsize'] = [8, 8]


def train_and_save_model(data_file, checkpoint_file):
    # Load the data from the CSV file
    data = pd.read_csv(data_file)

    # Remove the car_ID column
    data = data.drop('car_ID', axis=1)

    # Check for missing values
    missing_vals = check_missing_values(data)
    print(missing_vals.head())

    # Clean the car names
    unique_car_names = clean_car_names(data)
    print(unique_car_names)

    # Convert the symboling column to a string
    convert_symboling_to_string(data)

    # Select the best features
    num_cols = data.select_dtypes(exclude=['object']).columns
    best_features = select_features(data, num_cols)
    print(best_features)
    feature_ranges = {}
    for feature in best_features:
        feature_data = data[feature]
        feature_range = (feature_data.min(), feature_data.max())
        feature_ranges[feature] = feature_range
        return feature_ranges

    # Split the data into input (X) and target (y)
    X = data[best_features]
    y = data['price']

    # Train a GradientBoostingRegressor model
    model = GradientBoostingRegressor()
    model.fit(X, y)

    # Save the trained model as a checkpoint file
    joblib.dump(model, checkpoint_file)
    


def predict_price(wheelbase, carlength, carwidth, carheight, curbweight, enginesize, 
boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg):
    # Load the trained model checkpoint file
    model = joblib.load('/Users/diana/Desktop/MLOps/inference/model_checkpoint2.joblib')

    # Create a dataframe with the input features
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

    return prediction[0]


if __name__ == "__main__":
    data_file = '/Users/diana/Desktop/MLOps/data/CarPrice_Assignment.csv'
    checkpoint_file = '/Users/diana/Desktop/MLOps/inference/model_checkpoint2.joblib'

    train_and_save_model(data_file, checkpoint_file)
    price = predict_price(wheelbase = 88.6, carlength = 168.8, carwidth = 64.1, carheight = 48.8, curbweight = 2548, enginesize = 130, 
boreratio = 3.47, stroke = 2.68, compressionratio = 9, horsepower = 111, peakrpm = 5000, citympg = 21, highwaympg = 27)
    print('Predicted price:', price)
    
