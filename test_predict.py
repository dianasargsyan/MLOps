import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

from utils import check_missing_values
from utils import convert_symboling_to_string
from utils import select_features
from utils import clean_car_names
from predict import train_and_save_model


data=pd.read_csv('CarPrice_Assignment.csv')
data = data.drop('car_ID',axis=1)
num_cols=data.select_dtypes(exclude=['object']).columns

def test_check_missing_values_no_nans():
    # Call the function
    result = check_missing_values(data)
    
    # Assert that there are no NaN values in the output
    assert not pd.isna(result).any()
def test_clean_car_names():
    expected_output = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi', 'Nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
    assert clean_car_names(data) == expected_output


def test_convert_symboling_to_string():
    convert_symboling_to_string(data)
    assert data['symboling'].dtype == 'object'
   
def test_select_features_length():
    # call the select_features function
    selected_features = select_features(data, num_cols)
    
    # check that the function returns a list of 13 features
    assert len(selected_features) == 13
def test_select_features_numcols():
    # call the select_features function
    selected_features = select_features(data, num_cols)
        
    # check that all the selected features are in the original set of numerical columns
    assert set(selected_features).issubset(set(num_cols))

def test_select_features_dfcols():
    # call the select_features function
    selected_features = select_features(data, num_cols)
    
    # check that all the selected features are in the dataframe's columns
    assert set(selected_features).issubset(set(data.columns))
def test_select_features_validcols():
    # call the select_features function
    selected_features = select_features(data, num_cols)
        
    # check that the selected features are valid column names in the dataframe
    assert all(col in data.columns for col in selected_features)
    
def test_select_features_price():
    # call the select_features function
    selected_features = select_features(data, num_cols)
    # check that the selected features are not the target column 'price'
    assert 'price' not in selected_features
