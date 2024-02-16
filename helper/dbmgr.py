from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
from helper.client import *
from helper.machinelearner import *
from functools import partial
from pymongo import *
import pymongo

class DatabaseManager:
    def __init__(self, db_uri='mongodb://localhost:27017/', db_name='crypto_data', start_time=None, end_time=None):
        self.db_uri = db_uri  # Store URI for use in worker functions
        self.db_name = db_name
        self.start_time = start_time
        self.end_time = end_time
        self.granularities = [
            "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE",
            "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"
        ]
    
    @staticmethod
    def generate_expected_timestamps(start_time, end_time, granularity):
        granularity_mapping = {
            'ONE_MINUTE': 1,
            'FIVE_MINUTE': 5,
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
            "ONE_MINUTE": timedelta(minutes=1),
            "FIVE_MINUTE": timedelta(minutes=5),
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
            'ONE_MINUTE': 1,
            'FIVE_MINUTE': 5,
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

        def identify_gaps(missing_timestamps):
            gaps = []
            gap_start = missing_timestamps[0]

            for i in range(1, len(missing_timestamps)):
                if missing_timestamps[i] - missing_timestamps[i-1] > timedelta(minutes=granularity_mapping[granularity]):
                    gap_end = missing_timestamps[i-1]
                    gaps.append((gap_start, gap_end))
                    gap_start = missing_timestamps[i]

            # Add the last gap
            gaps.append((gap_start, missing_timestamps[-1]))
            return gaps

        # Generate expected timestamps
        expected_timestamps = list(DatabaseManager.generate_expected_timestamps(start_time, end_time, granularity))

        # Find missing timestamps
        existing_documents = collection.find(
            {'instrument': pid, 'start': {'$gte': start_time, '$lt': end_time}},
            {'start': 1, '_id': 0}
        )
        existing_timestamps = [doc['start'] for doc in existing_documents]
        missing_timestamps = sorted(set(expected_timestamps) - set(existing_timestamps))

        # Identify gaps (missing periods) and fetch data for each gap
        gaps = identify_gaps(missing_timestamps)
        for gap in tqdm(gaps, desc='Fetching data for gaps'):
            gap_start, gap_end = gap
            df = CoinbaseClient(api_key, api_secret).get_product_candles(pid, gap_start, gap_end, granularity)

        if missing_timestamps:
            # Determine the range for missing data
            missing_start = missing_timestamps[0]
            missing_end = missing_timestamps[-1] + timedelta(minutes=granularity_mapping[granularity])

            # Fetch data for the range of missing timestamps
            df = CoinbaseClient(api_key, api_secret).get_product_candles(pid, missing_start, missing_end, granularity)

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
        
        # Fetch the last document before the missing period for forward filling
        last_doc = collection.find_one({'instrument': pid, 'start': {'$lt': missing_timestamps[0]}}, sort=[('start', pymongo.DESCENDING)])

        for missing_timestamp in missing_timestamps:
            # Create a new document with forward-filled price data and zero volume
            new_doc = {
                'instrument': pid,
                'granularity': granularity,
                'start': missing_timestamp,
                'end': missing_timestamp + timedelta(minutes=granularity_mapping[granularity]),
                'open': last_doc['open'],
                'close': last_doc['close'],
                'high': last_doc['high'],
                'low': last_doc['low'],
                'volume': 0
            }

            # Insert the new document into the collection
            collection.insert_one(new_doc)

            # Update the last_doc for the next iteration
            last_doc = new_doc
            
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