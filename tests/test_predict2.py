import pytest
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.utils import check_missing_values, convert_symboling_to_string, select_features, clean_car_names

warnings.filterwarnings("ignore")
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]

@pytest.fixture
def car_data():
    data = pd.read_csv('/Users/diana/Desktop/MLOps/data/CarPrice_Assignment.csv')
    data = data.drop('car_ID', axis=1)
    return data

def test_check_missing_values_no_nans(car_data):
    # Call the function
    result = check_missing_values(car_data)

    # Assert that there are no NaN values in the output
    assert not pd.isna(result).any()

def test_clean_car_names(car_data):
    expected_output = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi', 'Nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
    assert clean_car_names(car_data) == expected_output


def test_convert_symboling_to_string(car_data):
    convert_symboling_to_string(car_data)
    assert car_data['symboling'].dtype == 'object'

def test_select_features_length(car_data):
    # call the select_features function
    selected_features = select_features(car_data, car_data.select_dtypes(exclude=['object']).columns)

    # check that the function returns a list of 13 features
    assert len(selected_features) == 13

def test_select_features_numcols(car_data):
    # call the select_features function
    selected_features = select_features(car_data, car_data.select_dtypes(exclude=['object']).columns)

    # check that all the selected features are in the original set of numerical columns
    assert set(selected_features).issubset(set(car_data.select_dtypes(exclude=['object']).columns))

def test_select_features_dfcols(car_data):
    # call the select_features function
    selected_features = select_features(car_data, car_data.select_dtypes(exclude=['object']).columns)

    # check that all the selected features are in the dataframe's columns
    assert set(selected_features).issubset(set(car_data.columns))

def test_select_features_validcols(car_data):
    # call the select_features function
    selected_features = select_features(car_data, car_data.select_dtypes(exclude=['object']).columns)

    # check that the selected features are valid column names in the dataframe
    assert all(col in car_data.columns for col in selected_features)

def test_select_features_price(car_data):
    # call the select_features function
    selected_features = select_features(car_data, car_data.select_dtypes(exclude=['object']).columns)
    # check that the selected features are not the target column 'price'
    assert 'price' not in selected_features
