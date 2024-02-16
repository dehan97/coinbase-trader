from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import yfinance as yf
import ta

class FeatureEngineer:
    def __init__(self, db_uri='mongodb://localhost:27017/', db_name='crypto_data'):
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]

    def fetch_financial_data(self, ticker_symbol, start_date, end_date):
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date, end=end_date, interval="1D")
        return pd.DataFrame(data).reset_index()

    def fetch_table_data(self, collection_name):
        collection = self.db[collection_name]
        cursor = collection.find({}, {'_id': 0, 'Date': 1, 'Close': 1}).sort('Date', -1)
        df = pd.DataFrame(list(cursor))
        print(df)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        return df.set_index('Date')

    def calculate_differences_and_momentum(self, df, intervals):
        for interval in intervals:
            df[f'Momentum_{interval}d'] = df['Close'].diff(periods=interval) / df['Close'].shift(periods=interval)
            df[f'Momentum_ROC_{interval}d'] = df[f'Momentum_{interval}d'].diff() / interval
            df[f'Close_diff_{interval}d'] = df['Close'].diff(periods=interval)
            df[f'Close_diff_{interval}d_bps'] = df[f'Close_diff_{interval}d'] / df['Close'].shift(periods=interval) * 10000
        return df

    def create_breadth_df(self, intervals):
        table_names = ['BTC_daily', 'ETH_daily', 'SP500_daily', 'VIX_daily', 'Magnificent7_Index_daily']
        breadth_df = pd.DataFrame()

        for collection_name in table_names:
            df = self.fetch_table_data(collection_name)
            df = self.calculate_differences_and_momentum(df, intervals)
            for interval in intervals:
                breadth_df[f'{collection_name}_Momentum_{interval}d'] = df[f'Momentum_{interval}d']

        breadth_df.fillna(method='bfill', inplace=True)
        return breadth_df
    
    ### HIGH-LOW
    def calculate_btc_and_instrument_high_low(self, collection_name):
        collection = self.db[collection_name]
        btc_data = pd.DataFrame(list(collection.find({'instrument': 'BTC-USDC'})))
        instruments_data = pd.DataFrame(list(collection.find({'instrument': {'$ne': 'BTC-USDC'}})))

        for instrument_data in instruments_data:
            btc_high_low = next((data for data in btc_data if data['start'] == instrument_data['start']), None)
            if btc_high_low:
                high_btc = btc_high_low['high']
                low_btc = btc_high_low['low']
            else:
                high_btc = None
                low_btc = None

            record = {
                'start': instrument_data['start'],
                'instrument': instrument_data['instrument'],
                'high': instrument_data['high'],
                'low': instrument_data['low'],
                'high_btc': high_btc,
                'low_btc': low_btc,
                'end': instrument_data['end']
            }
            self.db['instr_high_low'].insert_one(record)

    ###
    def calculate_btc_and_instrument_volatility(self, collection_name, timeframes):
        collection = self.db[collection_name]
        btc_data = pd.DataFrame(list(collection.find({'instrument': 'BTC-USDC'})))
        other_instruments_data = pd.DataFrame(list(collection.find({'instrument': {'$ne': 'BTC-USDC'}})))

        # Prepare an empty DataFrame to store volatility results
        volatility_results = pd.DataFrame()

        # Loop over all timeframes to calculate volatility
        for timeframe in timeframes:
            # Calculate rolling standard deviation for BTC
            btc_data[f'volatility_{timeframe}d'] = btc_data['close'].rolling(window=timeframe, min_periods=1).std()

            # Calculate volatility for each other instrument and compare with BTC
            for index, row in other_instruments_data.iterrows():
                instrument_name = row['instrument']
                instrument_data = other_instruments_data[other_instruments_data['instrument'] == instrument_name]
                instrument_data[f'volatility_{timeframe}d'] = instrument_data['close'].rolling(window=timeframe, min_periods=1).std()

                # Compare instrument volatility with BTC and store in the results DataFrame
                instrument_data[f'volatility_vs_btc_{timeframe}d'] = instrument_data[f'volatility_{timeframe}d'] - btc_data.iloc[0][f'volatility_{timeframe}d']
                volatility_results = pd.concat([volatility_results, instrument_data])

        # Update MongoDB with the volatility results
        for index, row in volatility_results.iterrows():
            collection.update_one({'_id': row['_id']}, {'$set': row.to_dict()}, upsert=True)
    
    ###
    def calculate_momentum(self, data, timeframes, is_instr=True):
        for timeframe in timeframes:
            momentum_key = f'instr_momentum_{timeframe}min' if is_instr else f'btc_momentum_{timeframe}min'
            data['close'] = pd.to_numeric(data['close'])
            data[momentum_key] = data['close'].pct_change(periods=timeframe)
            data[momentum_key].fillna(method='bfill', inplace=True)
        return data

    def calculate_and_store_momentum(self, collection_name, timeframes):
        collection = self.db[collection_name]
        instruments_data = list(collection.find({}))

        for instrument_data in instruments_data:
            instrument_df = pd.DataFrame(instrument_data)
            for timeframe in timeframes:
                instrument_df = self.calculate_momentum(instrument_df, [timeframe])

                # Store the updated DataFrame back into MongoDB
                for index, row in instrument_df.iterrows():
                    collection.update_one({'_id': row['_id']}, {'$set': row.to_dict()})
                
    ###
    def convert_to_numeric(self, df, column_names):
        for column in column_names:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df

    def detect_crossover(self, data, column1, column2):
        crossover_up = (data[column1] > data[column2]) & (data[column1].shift() < data[column2].shift())
        crossover_down = (data[column1] < data[column2]) & (data[column1].shift() > data[column2].shift())
        data[f'{column1}_cross_{column2}_up'] = crossover_up
        data[f'{column1}_cross_{column2}_down'] = crossover_down

    def calculate_indicators(self, data, max_window):
        window = min(len(data), max_window)
        numeric_columns = ['open', 'high', 'low', 'close']
        data = self.convert_to_numeric(data, numeric_columns)

        # Calculate technical indicators using 'ta' library
        data['bb_bbm'] = ta.volatility.bollinger_mavg(data['close'], window=window, fillna=True)
        data['bb_bbh'] = ta.volatility.bollinger_hband(data['close'], window=window, fillna=True)
        data['bb_bbl'] = ta.volatility.bollinger_lband(data['close'], window=window, fillna=True)
        data['sma'] = ta.trend.sma_indicator(data['close'], window=window, fillna=True)
        data['ema'] = ta.trend.ema_indicator(data['close'], window=window, fillna=True)
        data['rsi'] = ta.momentum.rsi(data['close'], window=window, fillna=True)
        data['macd'] = ta.trend.macd(data['close'], fillna=True)
        data['macd_signal'] = ta.trend.macd_signal(data['close'], fillna=True)
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(high=data['high'], low=data['low'], window1=window, window2=window*2, window3=window*4)
        data[f'ichimoku_a_{window}'] = ichimoku.ichimoku_a()
        data[f'ichimoku_b_{window}'] = ichimoku.ichimoku_b()
        data[f'ichimoku_base_line_{window}'] = ichimoku.ichimoku_base_line()
        data[f'ichimoku_conversion_line_{window}'] = ichimoku.ichimoku_conversion_line()

        # Calculate percentage distance for each indicator
        for col in ['bb_bbm', 'bb_bbh', 'bb_bbl', 'sma', 'ema', 'rsi', 'macd', 'macd_signal']:
            data[f'{col}_pct_dist'] = (data[col] - data['close']) / data['close'] * 100

        # Detect crossovers
        self.detect_crossover(data, 'macd', 'macd_signal')

        return data

    def calculate_indicators_for_all_instruments_and_granularities(self, max_window):
        results = []

        # Assuming you have a collection per instrument with documents containing 'granularity', 'start', and other required fields
        for collection_name in self.db.list_collection_names():
            collection = self.db[collection_name]
            cursor = collection.find()
            df = pd.DataFrame(list(cursor))

            # Ensure data is sorted and indexed by 'start'
            df.sort_values(by='start', ascending=True, inplace=True)
            df.set_index('start', inplace=True)

            # Convert necessary columns to numeric
            df = self.convert_to_numeric(df, ['open', 'high', 'low', 'close'])

            # Calculate indicators
            df_with_indicators = self.calculate_indicators(df, max_window)

            # Add instrument and granularity info
            df_with_indicators['instrument'] = collection_name
            df_with_indicators['granularity'] = df_with_indicators.apply(lambda x: self.get_granularity_from_collection(collection_name), axis=1)

            results.append(df_with_indicators)

        # Concatenate all results into a single DataFrame
        final_data = pd.concat(results, ignore_index=True)

        # You might want to save the processed data into a new collection in MongoDB
        self.save_to_mongodb(final_data, 'processed_indicators')

    def run_feature_engineering(self):
        # Define the intervals for breadth_df calculation
        intervals = [1, 2, 3, 7, 14, 21, 30]
        # Calculate the breadth_df using MongoDB data
        breadth_df = self.create_breadth_df(intervals)
        print(breadth_df.head())
        # Store breadth_df in a MongoDB collection if needed
        breadth_collection = self.db['breadth_data']
        records = breadth_df.to_dict('records')
        breadth_collection.insert_many(records)

        # Define timeframes for volatility and momentum calculations
        timeframes = [1, 10, 15, 100, 300, 1000]  # Timeframes in days for volatility
        self.calculate_btc_and_instrument_volatility('crypto_candles', timeframes)

        timeframes = [1, 5, 7, 10, 14, 21, 30, 69, 90, 300, 1337]  # Timeframes for momentum
        self.calculate_and_store_momentum('crypto_candles', timeframes)

        # Define the maximum window for technical indicators calculation
        max_window = 15*60  # Example with a 14-period window
        # Calculate indicators for all instruments and granularities using MongoDB data
        indicators_data = self.calculate_indicators_for_all_instruments_and_granularities('crypto_candles', max_window)
        print(indicators_data)
        # Optionally, store the indicators_data in MongoDB
        indicators_collection = self.db['instrument_indicators']
        indicators_records = indicators_data.to_dict('records')
        indicators_collection.insert_many(indicators_records)

    def __del__(self):
        self.client.close()
    