# Description
This platform is designed for cryptocurrency enthusiasts and data scientists interested in analyzing cryptocurrency market trends and developing trading strategies. It leverages the Coinbase API for real-time and historical market data, processes this data to compute various technical indicators, and uses machine learning algorithms to predict market movements. The platform also includes a backtesting framework to evaluate the performance of trading strategies.

# Components
client.py: Interfaces with the Coinbase API to fetch cryptocurrency market data.
dataprocessor.py: Processes raw market data to compute technical indicators and features for analysis.
dbmgr.py: Manages a MongoDB database for storing and retrieving processed market data.
machinelearner.py: Implements machine learning algorithms for predicting cryptocurrency price movements.
backtester_pt1.ipynb: A Jupyter notebook for backtesting trading strategies against historical data.
