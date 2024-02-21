from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
from helper.client import *
from helper.machinelearner import *
from functools import partial
from pymongo import *
import pymongo
from tqdm import tqdm  

class DatabaseManager:
    def __init__(self, db_uri='mongodb://localhost:27017/', db_name='crypto_data', start_time=None, end_time=None):
        self.db_uri = db_uri  # Store URI for use in worker functions
        self.db_name = db_name
        self.start_time = start_time
        self.end_time = end_time
        self.granularities = [
            "FIFTEEN_MINUTE", "THIRTY_MINUTE",
            "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"
        ]
    
    @staticmethod
    def generate_expected_timestamps(start_time, end_time, granularity):
        granularity_mapping = {
            'FIFTEEN_MINUTE': 15,
            'THIRTY_MINUTE': 30,
            'ONE_HOUR': 60,
            'TWO_HOUR': 120,
            'SIX_HOUR': 360,
            'ONE_DAY': 1440
        }
        delta = timedelta(minutes=granularity_mapping[granularity])
        current_time = start_time
        while current_time < end_time:
            yield current_time
            current_time += delta
            
    @staticmethod
    def calculate_end_time(start, granularity):
        granularities = {
            "FIFTEEN_MINUTE": timedelta(minutes=15),
            "THIRTY_MINUTE": timedelta(minutes=30),
            "ONE_HOUR": timedelta(hours=1),
            "TWO_HOUR": timedelta(hours=2),
            "SIX_HOUR": timedelta(hours=6),
            "ONE_DAY": timedelta(days=1)
        }
        return start + granularities.get(granularity, timedelta(0))

    def fetch_and_store_data(pid, granularity, db_name, db_uri, start_time, end_time):
        granularity_mapping = {
            'FIFTEEN_MINUTE': 15,
            'THIRTY_MINUTE': 30,
            'ONE_HOUR': 60,
            'TWO_HOUR': 120,
            'SIX_HOUR': 360,
            'ONE_DAY': 1440
        }
        client = MongoClient(db_uri)
        db = client[db_name]
        collection = db[f"crypto_candles_{granularity.lower()}"]

        # Ensure the collection has the required index
        collection.create_index([('instrument', pymongo.ASCENDING), ('granularity', pymongo.ASCENDING), ('start', pymongo.ASCENDING)], unique=True)

        # Fetch data for the entire time period without checking for missing timestamps
        df = CoinbaseClient(api_key, api_secret).get_product_candles(pid, start_time, end_time, granularity)

        if not df.empty:
            # Process and store each candle
            for _, row in df.iterrows():
                record = {
                    'instrument': pid,
                    'granularity': granularity,
                    'start': row['start'],
                    'end': row['start'] + timedelta(minutes=granularity_mapping[granularity]),
                    'open': row['open'],
                    'close': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume']
                }
                collection.update_one(
                    {'instrument': pid, 'start': record['start']},
                    {'$set': record},
                    upsert=True
                )
        client.close()


    def main(self, unique_pids):
        print("Preparing to fetch and store data...")
        # Prepare tasks as tuples containing all necessary arguments for fetch_and_store_data
        tasks = [(pid, granularity, self.db_name, self.db_uri, self.start_time, self.end_time) for pid in unique_pids for granularity in self.granularities]
        
        with Pool(processes=30) as pool:
            # Since each task is a tuple containing all arguments, use starmap to unpack them when calling the function
            for _ in tqdm(pool.starmap(DatabaseManager.fetch_and_store_data, tasks), total=len(tasks), desc="Processing"):
                pass

        print("All data fetching tasks completed.")

    def check_row_counts(self):
        client = MongoClient(self.db_uri)
        db = client[self.db_name]
        print("Checking the number of documents in each collection...")
        for granularity in self.granularities:
            collection_name = f"crypto_candles_{granularity.lower()}"
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"Collection '{collection_name}' has {count} documents.")
        client.close()