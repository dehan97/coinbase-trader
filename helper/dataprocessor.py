from pymongo import MongoClient
import pandas as pd
import ta
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pymongo import UpdateOne
import os

class FeatureEngineer:
    """
    For each tick granularity, and for each instrument, calculate technical indicators:
    1. 5 day moving average of price
    2. 20 days moving average of high and low
    3. RSI
    4. On Balance volume
    5. MACD
    6. Rate of change
    7. Stochasstic oscillator
    8. Commodity channel index

    Where appropriate, calculate the percentage distance away from the indicator instead of an absolute value
    """
    def __init__(self, db_name):
        self.client = MongoClient()
        self.db = self.client[db_name]
        self.master_collection = 'crypto_candles_one_minute'
        self.pool_size = 4
        self.collection_names = [
            'crypto_candles_fifteen_minute',
            'crypto_candles_one_day',
            'crypto_candles_one_hour',
            'crypto_candles_six_hour',
            'crypto_candles_thirty_minute',
            'crypto_candles_two_hour'
        ]

    def get_master_list(self):
        collection = self.db[self.master_collection]
        master_data = pd.DataFrame(list(collection.find({}, {'start': 1, 'instrument': 1})))
        return master_data.drop_duplicates()
    
    def fetch_data(self, collection_name):
        collection = self.db[collection_name]
        data = pd.DataFrame(list(collection.find()))
        numeric_cols = ['close', 'high', 'low', 'volume']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        return data

    def calculate_group_indicators(self, group):
        window = 30  # Define the rolling window size for indicators that need it

        # Moving averages
        group['9d_ma'] = group.groupby('instrument')['close'].transform(lambda x: x.rolling(window=9).mean())
        group['20d_ma_high'] = group.groupby('instrument')['high'].transform(lambda x: x.rolling(window=20).mean())
        group['20d_ma_low'] = group.groupby('instrument')['low'].transform(lambda x: x.rolling(window=20).mean())

        # RSI - Relative Strength Index
        group['rsi'] = ta.momentum.RSIIndicator(group['close']).rsi()

        # OBV - On Balance Volume
        group['obv'] = ta.volume.OnBalanceVolumeIndicator(group['close'], group['volume']).on_balance_volume()

        # ADL - Accumulation/Distribution Line
        group['adl'] = ta.volume.AccDistIndexIndicator(group['high'], group['low'], group['close'], group['volume']).acc_dist_index()

        # VWAP - Volume-Weighted Average Price (note: typically requires intraday data)
        group['price_volume'] = group['close'] * group['volume']

        # Extract date from 'start' timestamp
        group['date'] = pd.to_datetime(group['start']).dt.date

        volume_per_day = group.groupby(['date', 'instrument'])['volume'].transform('sum')
        pv_sum_per_day = group.groupby(['date', 'instrument'])['price_volume'].transform('sum')
        group['vwap'] = pv_sum_per_day / volume_per_day
        

        # CMF - Chaikin Money Flow
        group['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(group['high'], group['low'], group['close'], group['volume'], window=20).chaikin_money_flow()

        # MACD - Moving Average Convergence Divergence
        macd = ta.trend.MACD(group['close'])
        group['macd'] = macd.macd()
        group['macd_signal'] = macd.macd_signal()

        # ROC - Rate of Change
        group['roc'] = ta.momentum.ROCIndicator(group['close']).roc()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(group['high'], group['low'], group['close'])
        group['stoch_k'] = stoch.stoch()
        group['stoch_d'] = stoch.stoch_signal()

        # CCI - Commodity Channel Index
        group['cci'] = ta.trend.CCIIndicator(group['high'], group['low'], group['close']).cci()

        return group.drop(columns=['date'])


    def calculate_indicators(self, df):
        # Apply calculate_group_indicators to each group and combine the results
        processed_df = df.groupby('instrument').apply(self.calculate_group_indicators).reset_index(drop=True)

        return processed_df

    def fetch_and_process_data(self, collection_name):
        granularity = collection_name.split('_')[-2]  # Extract granularity from collection name
        data = self.fetch_data(collection_name)
        data['granularity'] = granularity  # Add granularity as a column

        processed_data = self.calculate_indicators(data)
        return processed_data

    def process_and_aggregate(self, master_list):
        aggregated_data = pd.DataFrame()
        for collection_name in tqdm(self.collection_names, desc='Processing Collections'):
            data = self.fetch_and_process_data(collection_name)
            if aggregated_data.empty:
                aggregated_data = data
            else:
                aggregated_data = pd.concat([aggregated_data, data], ignore_index=True)
        
        # Ensure 'start', 'instrument', and 'granularity' are columns, not indices
        aggregated_data.reset_index(inplace=True)

        return aggregated_data

    def store_in_mongodb(self, processed_data, collection_name='crypto_TA'):
        collection = self.db[collection_name]

        # Ensure the DataFrame has the required columns
        required_columns = ['start', 'instrument', 'granularity']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in processed_data: {missing_columns}")
        
        # Convert DataFrame to list of dictionaries
        records = processed_data.to_dict('records')

        updates = []
        for record in records:
            # Exclude '_id' field from the record if it exists
            record.pop('_id', None)
            record.pop('index', None)
            
            filter_ = {'start': record['start'], 'instrument': record['instrument'], 'granularity': record['granularity']}
            updates.append(UpdateOne(filter_, {'$set': record}, upsert=True))

        if updates:
            collection.bulk_write(updates)

    def run(self):
        master_list = self.get_master_list()
        aggregated_data = self.process_and_aggregate(master_list)
        self.store_in_mongodb(aggregated_data)
        print("Aggregated data stored in 'crypto_TA' collection.")


class DataPreparator(FeatureEngineer):
    def __init__(self, db_name):
        super().__init__(db_name)

    def retrieve_and_prepare_data(self):
        fifteen_min_candles = self.fetch_data('crypto_candles_fifteen_minute')
        fifteen_min_candles.fillna(method='bfill', inplace=True)
        fifteen_min_candles.fillna(method='ffill', inplace=True)

        final_data = fifteen_min_candles

        # Dynamically get unique granularities from crypto_TA
        granularities = self.db['crypto_TA'].distinct('granularity')

        for granularity in granularities:
            ta_data = self.db['crypto_TA'].find({'granularity': granularity})
            ta_df = pd.DataFrame(list(ta_data))

            # Drop the '_id' column from ta_df to prevent merge conflicts
            ta_df.drop(columns=['_id', 'volume'], inplace=True, errors='ignore')

            # Rename TA indicator columns to include granularity, and keep 'start', 'instrument'
            ta_df.rename(columns=lambda x: f"{x}_{granularity}" if x not in ['start', 'instrument'] else x, inplace=True)

            ta_df.fillna(method='bfill', inplace=True)
            ta_df.fillna(method='ffill', inplace=True)

            # Merge using custom suffixes to handle any remaining duplicate columns
            final_data = pd.merge(final_data, ta_df, on=['start', 'instrument'], how='left', suffixes=('', f'_{granularity}'))
        
        # Check for columns with nulls and drop them
        null_columns = final_data.columns[final_data.isnull().any()]
        final_data.drop(null_columns, axis=1, inplace=True)

        # Log dropped columns
        logs_dir = 'Logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        with open(os.path.join(logs_dir, 'dropped_columns_log.txt'), 'w') as log_file:
            log_file.write("Dropped columns due to nulls:\n")
            log_file.write("\n".join(null_columns))

        for record in final_data.to_dict('records'):
            # Use the combination of fields that uniquely identify each document, such as 'instrument' and 'start'
            filter_ = {'instrument': record['instrument'], 'start': record['start']}
            update = {'$set': record}
            self.db['ML_features'].update_one(filter_, update, upsert=True)

        return final_data

    # Additional methods for labeling, scaling, and checking stationarity 
    def scale_data(self, df):
        scaler = MinMaxScaler()
        instruments = df['instrument'].unique()  # Assuming 'instrument' column identifies different instruments
        
        for instrument in instruments:
            instrument_index = df[df['instrument'] == instrument].index  # Get the index of rows for the current instrument
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Scale numerical features for the current instrument
            scaled_values = scaler.fit_transform(df.loc[instrument_index, numerical_features])
            
            # Assign the scaled values back using .loc to ensure direct assignment
            df.loc[instrument_index, numerical_features] = scaled_values

        return df

    def check_stationarity(self, series, significance_level=0.01):
        result = adfuller(series.dropna())
        if result[1] < significance_level:
            return True  # Stationary
        else:
            return False  # Not stationary

    def ensure_stationarity(self, df):
        non_stationary_columns = []
        report = []

        for column in df.select_dtypes(include=[np.number]).columns:
            stationary = self.check_stationarity(df[column])
            if not stationary:
                non_stationary_columns.append(column)
                report.append(f"Column {column} is non-stationary.")
            else:
                report.append(f"Column {column} is stationary.")

        # Ensure 'Logs' directory exists
        logs_dir = 'Logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Write the report to a text file
        with open(os.path.join(logs_dir, 'stationarity_report.txt'), 'w') as file:
            for line in report:
                file.write(line + '\n')

        return df, non_stationary_columns

    def prepare_data(self):
        df = self.retrieve_and_prepare_data()
        df_scaled = self.scale_data(df)
        df_final, non_stationary_columns = self.ensure_stationarity(df_scaled)
        return df_final, non_stationary_columns

###
class LabelGenerator:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', 
                 db_name='crypto_data', 
                 account_value=5000, tx_fee=0.006, profit_level=0.003):
        self.client = MongoClient(mongo_uri)
        self.db_name = db_name
        self.db = self.client[db_name]
        self.account_value = account_value
        self.tx_fee = tx_fee
        self.profit_level = profit_level
        self.granularities = [
            "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE",
            "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"
        ]

    def get_instruments(self):
        collection = self.db[self.db_name]
        instruments = collection.distinct("instrument", {"granularity": "ONE_MINUTE"})
        return instruments

    def fetch_data(self, instrument):
        collection = self.db[self.db_name]
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
    
    def run(self):
        collection_name = 'ML_features'  # The collection to save labeled data
        instruments = self.get_instruments()

        for instrument in tqdm(instruments):
            df = self.fetch_data(instrument)

            if not df.empty:
                for granularity in self.granularities:
                    df_granularity = df[df['granularity'] == granularity]

                    for window_size in [15, 30, 45, 60]:
                        if not df_granularity.empty:
                            result_df = self.calculate_targets(df_granularity.copy(), window_size)
                            result_df['instrument'] = instrument
                            result_df['granularity'] = granularity
                            result_df['window_size'] = window_size

                            self.save_to_database(result_df, collection_name)
                            print(f"Saved labeled data for {instrument} with granularity {granularity} and window_size {window_size} to {collection_name}.")

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
