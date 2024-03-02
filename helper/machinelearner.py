from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb

class MongoDBMLModelTrainer:
    def __init__(self, mongo_uri, db_name, collection_name, target_column):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection_name = collection_name
        self.target_column = target_column

    def fetch_data(self):
        collection = self.db[self.collection_name]
        cursor = collection.find({})
        df = pd.DataFrame(list(cursor))
        # Assuming 'time' is the column used for temporal splitting; adjust if necessary
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        else:
            raise ValueError("The DataFrame does not contain a 'time' column.")
        return df

    def split_data(self, df):
        df.sort_values(by='time', inplace=True)
        validation_size = int(len(df) * 0.1)
        test_size = int(len(df) * 0.2)
        validation_set = df[:validation_size]
        test_set = df[-test_size:]
        train_set = df[validation_size:-test_size]
        X_train = train_set.drop(columns=[self.target_column, 'time'])
        y_train = train_set[self.target_column]
        X_validation = validation_set.drop(columns=[self.target_column, 'time'])
        y_validation = validation_set[self.target_column]
        X_test = test_set.drop(columns=[self.target_column, 'time'])
        y_test = test_set[self.target_column]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def train_and_evaluate(self):
        df = self.fetch_data()
        X_train, X_validation, X_test, y_train, y_validation, y_test = self.split_data(df)
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_validation)
            val_accuracy = accuracy_score(y_validation, val_predictions)
            print(f"{name} Validation Accuracy: {val_accuracy}")
            test_predictions = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_predictions)
            print(f"{name} Test Accuracy: {test_accuracy}")
            print(classification_report(y_test, test_predictions))

