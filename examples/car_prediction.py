import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils import test_watermark

from warnings import simplefilter
import random

random.seed(10)
simplefilter(action='ignore', category=UserWarning)


if __name__ == '__main__':

    # Load regression data
    df = pd.read_csv('data/car_data.csv')
    df = df[['year', 'selling_price', 'km_driven', 'fuel',
            'seller_type', 'transmission', 'owner']]
    # Create new features
    df['current_year'] = 2021
    df['years_old'] = df['current_year'] - df['year']
    df.drop(['year'], axis=1, inplace=True)
    df.drop(['current_year'], axis=1, inplace=True)
    # Convert categorical variable into dummy
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('selling_price', axis=1).copy()
    y = df['selling_price'].copy().values

    # Random Forest Regressor
    print('\n\nRandom Forest Regressor\n')
    base_model = RandomForestRegressor(max_depth=1000, random_state=10)
    test_watermark(X, y, base_model, metric='RMSE', trigger_size=5)
    test_watermark(X, y, base_model, metric='MAPE', trigger_size=5)
