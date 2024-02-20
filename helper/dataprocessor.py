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

from pymongo import MongoClient
import pandas as pd
import ta
from tqdm import tqdm

class FeatureEngineer:
    def __init__(self, db_name):
        self.client = MongoClient()
        self.db = self.client[db_name]
        self.master_collection = 'crypto_candles_one_minute'
        self.collection_names = [
            'crypto_candles_fifteen_minute',
            'crypto_candles_five_minute',
            'crypto_candles_one_day',
            'crypto_candles_one_hour',
            'crypto_candles_one_minute',
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

    def calculate_indicators(self, df):
        def calculate_group_indicators(group):
            group['5d_ma'] = group['close'].rolling(window=5).mean()
            group['20d_ma_high'] = group['high'].rolling(window=20).mean()
            group['20d_ma_low'] = group['low'].rolling(window=20).mean()
            group['rsi'] = ta.momentum.RSIIndicator(group['close']).rsi()
            group['obv'] = ta.volume.OnBalanceVolumeIndicator(group['close'], group['volume']).on_balance_volume()
            macd = ta.trend.MACD(group['close'])
            group['macd'] = macd.macd()
            group['macd_signal'] = macd.macd_signal()
            group['roc'] = ta.momentum.ROCIndicator(group['close']).roc()
            stoch = ta.momentum.StochasticOscillator(group['high'], group['low'], group['close'])
            group['stoch_k'] = stoch.stoch()
            group['stoch_d'] = stoch.stoch_signal()
            group['cci'] = ta.trend.CCIIndicator(group['high'], group['low'], group['close']).cci()
            return group

        return df.groupby('instrument').apply(calculate_group_indicators)

    def fetch_and_process_data(self, collection_name):
        granularity = collection_name.split('_')[-2]  # Extract granularity from collection name
        data = self.fetch_data(collection_name)
        data['granularity'] = granularity  # Add granularity as a column
        processed_data = self.calculate_indicators(data)
        return processed_data

    def process_and_aggregate(self, master_list):
        aggregated_data = pd.DataFrame()  # Initialize empty DataFrame for aggregated data
        for collection_name in tqdm(self.collection_names, desc='Processing Collections'):
            data = self.fetch_and_process_data(collection_name)
            if aggregated_data.empty:
                aggregated_data = data
            else:
                aggregated_data = pd.concat([aggregated_data, data], ignore_index=True)
        
        # After all data is aggregated, set 'start', 'instrument', and 'granularity' as index
        aggregated_data.set_index(['start', 'instrument', 'granularity'], inplace=True)
        
        return aggregated_data

    def store_in_mongodb(self, processed_data, collection_name='crypto_TA'):
        collection = self.db[collection_name]
        processed_data.reset_index(inplace=True)  # Reset index to store 'start', 'instrument', and 'granularity' as fields
        records = processed_data.to_dict('records')
        for record in records:
            filter_ = {'start': record['start'], 'instrument': record['instrument'], 'granularity': record['granularity']}
            update = {'$set': record}
            collection.update_one(filter_, update, upsert=True)

    def run(self):
        master_list = self.get_master_list()
        aggregated_data = self.process_and_aggregate(master_list)
        self.store_in_mongodb(aggregated_data)
        print("Aggregated data stored in 'crypto_TA' collection.")
