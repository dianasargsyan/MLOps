import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

pd.options.display.float_format = '{:.4f}'.format
plt.rcParams['figure.figsize'] = [8, 8]


def check_missing_values(data):
    """
    Returns a sorted pandas series containing the percentage of missing values
    for each column in the input dataframe.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to check for missing values.

    Returns:
    --------
    pandas.Series
        A pandas series containing the percentage of missing values for each
        column, sorted in descending order.
    """
    df_na = data.isna().mean().round(4) * 100
    return df_na.sort_values(ascending=False)


def clean_car_names(data):
    """
    Cleans up the 'CarName' column in the input dataframe by removing any null values,
    extracting the first word of each name, and returning a list of unique car names.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the 'CarName' column to clean up.

    Returns:
    --------
    list
        A list of unique car names extracted from the 'CarName' column after cleaning.
    """
    # Check for and remove any null values
    data['CarName'].isnull().sum()

    # Extract the first word from each car name
    data['CarName'] = data['CarName'].str.split(' ', expand=True)[0]

    # Return a list of unique car names
    data['CarName'].unique().tolist()
    data['CarName'] = data['CarName'].replace(
        {'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota',
         'vokswagen': 'volkswagen', 'vw': 'volkswagen'})
    return data['CarName'].unique().tolist()


def convert_symboling_to_string(data):
    """
    Converts the 'symboling' column in the input dataframe to a string data type.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the 'symboling' column to convert.

    Returns:
    --------
    None
    """
    data['symboling'] = data['symboling'].astype(str)


def select_features(data, num_cols):
    """
    Selects the best 15 features from the input dataframe's numerical columns using RFE and Gradient Boosting
    Regression.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the columns to be used for feature selection.
    num_cols: list
        A list of the numerical columns in the input dataframe.

    Returns:
    --------
    list
        A list of the 13 best features selected by RFE and Gradient Boosting Regression.
    """
    # Drop the 'price' column from the numerical columns and create a separate target variable y
    X = data[num_cols].drop('price', axis=1)
    y = data['price']

    # Apply label encoding to all columns in the dataframe
    X = X.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))

    # Drop the 'CarName' and 'price' columns from the encoded dataframe, if they exist
    if 'CarName' in X.columns:
        X = X.drop('CarName', axis=1)
    if 'price' in X.columns:
        X = X.drop('price', axis=1)

    # Perform RFE to select the best 15 features using Gradient Boosting Regression
    clf_rf_3 = GradientBoostingRegressor()
    rfe = RFE(estimator=clf_rf_3, n_features_to_select=13, step=1)
    rfe = rfe.fit(X, y)

    # Extract the names of the 15 best features
    features = list(X.columns[rfe.support_])

    # Return the list of best features
    return features


def gradient_boosting_regressor(X, y, num_estimators=15, random_state=20):
    """
    Trains and evaluates a Gradient Boosting Regressor on the input data and target variables.

    Parameters:
    -----------
    X : pandas.DataFrame
        The dataframe containing the features to be used for training and evaluation.
    y: pandas.Series
        The target variable for training and evaluation.
    num_estimators: int, optional (default=15)
        The number of estimators (trees) to use in the Gradient Boosting Regressor.
    random_state: int, optional (default=20)
        The random seed used for initialization of the Gradient Boosting Regressor.

    Returns:
    --------
    None
        Prints the train and test performance metrics of the trained Gradient Boosting Regressor.
    """
    # Select the best features using RFE
    clf_rf_3 = GradientBoostingRegressor()
    rfe = RFE(estimator=clf_rf_3, n_features_to_select=15, step=1)
    rfe = rfe.fit(X, y)

    # Extract the names of the 15 best features
    features = list(X.columns[rfe.support_])

    # Select the best features from the dataframe
    x = X[features]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    # Initialize the Gradient Boosting Regressor and fit the training data
    GB = GradientBoostingRegressor(n_estimators=num_estimators,
                                   random_state=random_state)
    GB.fit(x_train, y_train)

    # Make predictions on the training and testing data
    GB_train_pred = GB.predict(x_train)
    GB_test_pred = GB.predict(x_test)

    # Determine the performance metrics for the training data
    train_mse = mean_squared_error(y_train, GB_train_pred)
    train_rmse = mean_squared_error(y_train, GB_train_pred, squared=False)
    train_r2 = r2_score(y_train, GB_train_pred)

    # Print the performance metrics for the training data
    print(f'train mse: {int(train_mse)}')
    print(f'train rmse: {int(train_rmse)}')
    print(f'train r2: {train_r2}')
    print()

    # Determine the performance metrics for the testing data
    test_mse = mean_squared_error(y_test, GB_test_pred)
    test_rmse = mean_squared_error(y_test, GB_test_pred, squared=False)
    test_r2 = r2_score(y_test, GB_test_pred)

    # Print the performance metrics for the testing data
    print(f'test mse: {int(test_mse)}')
    print(f'test rmse: {int(test_rmse)}')
    print(f'test r2: {test_r2}')
    print()


