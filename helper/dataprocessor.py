from pymongo import MongoClient
import pandas as pd
import ta
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pymongo import UpdateOne

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
        #NOTE CALCULATE VOLUME INDICATORS
        # Calculate technical indicators for a given group (instrument)
        window = 30  # Define the rolling window size for indicators that need it

        # Moving averages
        group['9d_ma'] = group.groupby('instrument')['close'].transform(lambda x: x.rolling(window=9).mean())
        group['20d_ma_high'] = group.groupby('instrument')['high'].transform(lambda x: x.rolling(window=20).mean())
        group['20d_ma_low'] = group.groupby('instrument')['low'].transform(lambda x: x.rolling(window=20).mean())

        # RSI - Relative Strength Index
        group['rsi'] = ta.momentum.RSIIndicator(group['close']).rsi()

        # OBV - On Balance Volume
        group['obv'] = ta.volume.OnBalanceVolumeIndicator(group['close'], group['volume']).on_balance_volume()

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

        return group

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

        # Prepare bulk update operations
        updates = [
            UpdateOne(
                {'start': record['start'], 'instrument': record['instrument'], 'granularity': record['granularity']},
                {'$set': record},
                upsert=True
            )
            for record in records
        ]

        # Execute bulk update
        if updates:
            collection.bulk_write(updates)
        else:
            print("No updates to perform.")

    def run(self):
        master_list = self.get_master_list()
        aggregated_data = self.process_and_aggregate(master_list)
        self.store_in_mongodb(aggregated_data)
        print("Aggregated data stored in 'crypto_TA' collection.")


class DataPreparator(FeatureEngineer):
    def __init__(self, db_name):
        super().__init__(db_name)

    def retrieve_and_prepare_data(self):
        # Retrieve 15-minute candles
        fifteen_min_candles = self.fetch_data('crypto_candles_fifteen_minute')

        # Retrieve TA indicators for all granularities and join them
        ta_data = pd.DataFrame()
        for collection_name in self.collection_names:
            granularity_data = self.fetch_and_process_data(collection_name)
            if ta_data.empty:
                ta_data = granularity_data
            else:
                # Join on 'start' and 'instrument', ensuring alignment across different granularities
                ta_data = pd.merge(ta_data, granularity_data, on=['start', 'instrument'], how='left')

        # Merge 15-minute candles with TA indicators
        final_data = pd.merge(fifteen_min_candles, ta_data, on=['start', 'instrument'], how='left')

        return final_data

    # Additional methods for labeling, scaling, and ensuring stationarity would go here
    def scale_data(self, df):
        scaler = MinMaxScaler()
        instruments = df['instrument'].unique()  # Assuming 'instrument' column identifies different instruments
        for instrument in instruments:
            instrument_data = df[df['instrument'] == instrument]
            numerical_features = instrument_data.select_dtypes(include=[np.number]).columns.tolist()
            instrument_data[numerical_features] = scaler.fit_transform(instrument_data[numerical_features])
            df.update(instrument_data)
        return df
  
    def label_data(self, df, profit_threshold=0.01):
        # Assuming 'open' is the open price of the fifteen-minute candle and 'high' is the highest price in the next hour
        for index, row in df.iterrows():
            future_candles = df.iloc[index + 1 : index + 5]  # Next four candles for the next hour
            if not future_candles.empty:
                max_future_price = future_candles['high'].max()
                if (max_future_price - row['open']) / row['open'] >= profit_threshold:
                    df.at[index, 'label'] = 'BUY'
                else:
                    df.at[index, 'label'] = 'PASS'
        return df

    def check_stationarity(self, series, significance_level=0.05):
        result = adfuller(series.dropna())
        if result[1] < significance_level:
            return True  # Stationary
        else:
            return False  # Not stationary

    def ensure_stationarity(self, df):
        non_stationary_columns = []
        for column in df.columns:
            if not self.check_stationarity(df[column]):
                non_stationary_columns.append(column)
                # Implement logic to make the series stationary, e.g., differencing
        return df, non_stationary_columns

    def prepare_data(self):
        df = self.retrieve_and_prepare_data()
        df_labeled = self.label_data(df)
        df_scaled = self.scale_data(df_labeled)
        df_final, non_stationary_columns = self.ensure_stationarity(df_scaled)
        return df_final, non_stationary_columns
