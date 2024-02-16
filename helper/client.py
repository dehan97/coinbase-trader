import configparser
import requests
from requests.auth import AuthBase
import hmac
import hashlib
import time
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

config = configparser.ConfigParser()
config.read('config/config.ini')

api_key = config['coinbase']['api_key']
api_secret = config['coinbase']['api_secret']

class CBAuth(AuthBase):
    def __init__(self, secret, key, path):
        # setup any auth-related data here
        self.secret = secret
        self.key = key
        self.url = path

    def __call__(self, request):
        timestamp = str(int(time.time()))
        message = timestamp + request.method + self.url
        signature = hmac.new(self.secret.encode(
            'utf-8'), message.split('?')[0].encode('utf-8'), digestmod=hashlib.sha256).digest()

        request.headers.update({
            'CB-ACCESS-SIGN': signature.hex(),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.key,
            'accept': "application/json"
        })
        return request

class CoinbaseClient:
    def __init__(self, api_key, api_secret, base_url='https://api.coinbase.com'):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.granularity_to_seconds = {
                "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900, "THIRTY_MINUTE": 1800,
                "ONE_HOUR": 3600, "TWO_HOUR": 7200, "SIX_HOUR": 21600, "ONE_DAY": 86400
            }
        self.granularities = ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"]

    def auth(self, path):
        return CBAuth(self.api_secret, self.api_key, path)

    def get_accounts(self):
        path = '/api/v3/brokerage/accounts'
        url = self.base_url + path
        response = requests.get(url, auth=self.auth(path))
        return response.json()

    def get_products(self):
        path = '/api/v3/brokerage/products'
        url = self.base_url + path
        response = requests.get(url, auth=self.auth(path))
        return response.json()
    
    def get_unique_product_ids(self):
        products_json = self.get_products()
        product_ids = [product['product_id'] for product in products_json['products']]

        uniq_product_ids = set()
        for p in product_ids:
            if "USD" not in p.split("-")[0]:
                uniq_product_ids.add(p.split("-")[0])

        return uniq_product_ids

    def get_product_candles(self, ticker, start_time, end_time, granularity="FIFTEEN_MINUTE", max_retries=300):
        granularity_to_seconds = self.granularity_to_seconds
        candle_seconds = granularity_to_seconds[granularity]
        all_candles_df = pd.DataFrame()

        current_start = start_time
        while current_start < end_time:
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    current_end = min(current_start + timedelta(seconds=300 * candle_seconds), end_time)
                    chunk_candles = self._fetch_candles(ticker, current_start, current_end, granularity)
                    chunk_df = pd.json_normalize(chunk_candles['candles'])
                    all_candles_df = pd.concat([all_candles_df, chunk_df], ignore_index=True)
                    current_start = current_end
                    time.sleep(1/50)
                    break  # Break the retry loop on successful fetch
                except Exception as e:
                    print(f"Retry {retry_count}/{max_retries} for {ticker}, {current_start} to {current_end}, {granularity}: {e}")
                    time.sleep(10)  
                    retry_count += 1

            if retry_count > max_retries:
                print(f"Failed to fetch data after {max_retries} retries for {ticker}, {current_start} to {current_end}, {granularity}.")
                break  # Break the while loop if max retries exceeded

        if not all_candles_df.empty:
            all_candles_df['start'] = pd.to_datetime(all_candles_df['start'].astype(int), unit='s')
            all_candles_df['granularity'] = granularity
            all_candles_df['instrument'] = f"{ticker}"

        return all_candles_df

    def _fetch_candles(self, ticker, start_time, end_time, granularity):
        # Convert datetime objects to UNIX timestamps
        start_timestamp = str(int(time.mktime(start_time.timetuple())))
        end_timestamp = str(int(time.mktime(end_time.timetuple())))

        # Construct the request URL
        path = f"/api/v3/brokerage/products/{ticker}/candles?start={start_timestamp}&end={end_timestamp}&granularity={granularity}"
        url = self.base_url + path

        # Make the request and return the JSON response
        response = requests.get(url, auth=self.auth(path))
        if response.status_code != 200:
            print(f"Error fetching data: {response.json()}")
            return []
        return response.json()