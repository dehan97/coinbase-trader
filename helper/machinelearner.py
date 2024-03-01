import pandas as pd
import sqlite3
import concurrent.futures
from tqdm import tqdm
import numpy as np
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from itertools import product
from tqdm import tqdm
from pymongo import MongoClient

class LabelGenerator:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', 
                 db_name='crypto_data', 
                 account_value=5000, tx_fee=0.006, profit_level=0.003):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.account_value = account_value
        self.tx_fee = tx_fee
        self.profit_level = profit_level
        self.granularities = [
            "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE",
            "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"
        ]

    def get_instruments(self):
        collection = self.db['crypto_candles']
        instruments = collection.distinct("instrument", {"granularity": "ONE_MINUTE"})
        return instruments

    def fetch_data(self, instrument):
        collection = self.db['crypto_candles']
        cursor = collection.find({"instrument": instrument})
        df = pd.DataFrame(list(cursor))
        for column in ['close', 'high', 'low', 'volume']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df

    def calculate_time_to_profit(self, df, window_size, profit_level):
        if 'close' not in df.columns or 'high' not in df.columns:
            raise ValueError("DataFrame must contain 'close' and 'high' columns")

        close_prices = df['close'].values
        high_prices = df['high'].values
        target_prices = close_prices * (1 + profit_level)
        
        # Initialize an array to store the time to profit
        time_to_profit = np.full(len(close_prices), np.nan)

        # Iterate through the array
        for i in range(len(close_prices) - window_size):
            # Check if high price in the window reaches the target price
            target_reached = np.where(high_prices[i + 1:i + 1 + window_size] >= target_prices[i])[0]
            if target_reached.size > 0:
                time_to_profit[i] = target_reached[0] + 1  # +1 because we start checking from i+1

        return time_to_profit
    
    def save_to_csv(self, df, filename):
        os.makedirs('labels', exist_ok=True)
        filepath = os.path.join('labels', filename + '.csv')
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_from_csv(self, filename):
        filepath = os.path.join('labels', filename + '.csv')
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            return None
        
    def calculate_targets(self, df, window_size):
        # Calculate the maximum high and minimum low of the next 'window_size' candles
        next_high_max = f'next_{window_size}_high_max'
        next_low_min = f'next_{window_size}_low_min'
        df[next_high_max] = df['high'].rolling(window=window_size, min_periods=1).max().shift(-window_size + 1)
        df[next_low_min] = df['low'].rolling(window=window_size, min_periods=1).min().shift(-window_size + 1)

        # Calculate metrics over the previous window size
        window_open = f'window_{window_size}_open'
        window_high_max = f'window_{window_size}_high_max'
        window_low_min = f'window_{window_size}_low_min'
        window_volume_sum = f'window_{window_size}_volume_sum'

        df[window_open] = df['open'].shift(window_size - 1)
        df[window_high_max] = df['high'].rolling(window=window_size, min_periods=1).max().shift(1)
        df[window_low_min] = df['low'].rolling(window=window_size, min_periods=1).min().shift(1)
        df[window_volume_sum] = df['volume'].rolling(window=window_size, min_periods=1).sum().shift(1)

        # Get the latest valid close
        prev_close = f'prev_{window_size}_close'
        df[prev_close] = df['close'].shift(1)

        # Calculate potential maximum return and drawdown
        max_return = f'max_return_{window_size}'
        max_drawdown = f'max_drawdown_{window_size}'
        df[max_return] = (df[next_high_max] - df['close']) / df['close']
        df[max_drawdown] = (df['close'] - df[next_low_min]) / df['close']

        # Calculate threshold for 'BUY' based on fees and profit level
        threshold = 1 + 2 * self.tx_fee + self.profit_level

        # Label 'BUY' if condition is met
        target = f'target{window_size}'
        buy_condition = df[prev_close] * threshold < df[next_high_max]
        df[target] = np.where(buy_condition, 'BUY', 'HOLD')

        # Calculate time to reach the desired profit level
        time_col = f'time_to_profit_{window_size}min'
        df[time_col] = self.calculate_time_to_profit(df, window_size, self.profit_level)

        return df
    
    def generate_labels(self, chunk_size=1000000):
        instruments = self.get_instruments()

        for instrument in tqdm(instruments):
            # Fetch data from MongoDB collection for each instrument
            df = self.fetch_data(instrument)

            # Sort the DataFrame if necessary
            df = df.sort_values(by=['time'], ascending=True)  # Assume 'time' is the column for sorting
            unique_granularities = self.granularities

            for granularity in unique_granularities:
                df_granularity = df[df['granularity'] == granularity]

                for window_size in [15, 30, 45, 60]:
                    if not df_granularity.empty:
                        result_df = self.calculate_targets(df_granularity.copy(), window_size)
                        result_df['instrument'] = instrument
                        result_df['granularity'] = granularity
                        result_df['window_size'] = window_size
                        yield result_df

    def create_complete_index(self, df):
        all_start_dates = pd.date_range(start=df['start'].min(), end=df['start'].max(), freq='T')
        all_instruments = df['instrument'].unique()
        complete_index = pd.MultiIndex.from_product([all_start_dates, all_instruments], names=['start', 'instrument'])
        return complete_index
    
    def save_to_database(self, df, collection_name):
        collection = self.db[collection_name]
        # Convert DataFrame to dictionary format
        records = df.to_dict("records")
        collection.insert_many(records)
        print(f"Data saved to collection {collection_name} in MongoDB.")

    def load_from_database(self, collection_name):
        collection = self.db[collection_name]
        cursor = collection.find()
        df = pd.DataFrame(list(cursor))
        return df

    def data_exists_in_database(self, collection_name):
        collection_list = self.db.list_collection_names()  # Get the list of collection names
        exists = collection_name in collection_list  # Check if the collection name exists in the list
        return exists

    
###
#https://stackoverflow.com/questions/48034035/how-to-consistently-hot-encode-dataframes-with-changing-values
class MyOneHotEncoder():
    def __init__(self, all_possible_values):
        """
        all_possible_values: dict
            A dictionary where keys are column names and values are lists of all possible categories for each column.
        """
        self.all_possible_values = all_possible_values

    def transform(self, X, y=None):
        """
        Transforms the categorical columns in X using one-hot encoding.

        X: pandas DataFrame
            The data to transform.
        """
        # Get dummies for all columns in one go
        encoded = pd.get_dummies(X, prefix_sep='_')

        # Generate all possible column names
        all_possible_columns = self._generate_all_possible_column_names()

        # Reindex the encoded DataFrame to include all possible columns, filling missing with 0
        output_df = encoded.reindex(columns=all_possible_columns, fill_value=0)

        return output_df

    def _generate_all_possible_column_names(self):
        """
        Generates a list of all possible column names based on all_possible_values
        """
        all_columns = []
        for col, values in self.all_possible_values.items():
            all_columns.extend([f"{col}_{val}" for val in values])
        return all_columns
