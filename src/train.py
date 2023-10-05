import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import os
import sys


class ElasticNetModel:
    def __init__(self):
        self.rmse = 0
        self.mae = 0
        self.r2 = 0
        self.data_path = "/home/fero-jaxay/mlflow-demo/data/wine-quality.csv"


    def read_data(self):
        try:
            data = pd.read_csv(self.data_path, header=0)
            print("Data read Successfully")
            return data
        except Exception as e:
            print(e)

    def eval_metrics(self, real_data, predicted_data):
        self.rmse = np.sqrt(mean_squared_error(real_data, predicted_data))
        self.mae = mean_absolute_error(real_data, predicted_data)
        self.r2 = r2_score(real_data, predicted_data)
        return self.rmse, self.mae, self.r2

    def train_model(self, alpha, l1_ration):
        try:

            train_data = self.read_data()
            input_data = train_data.drop(['quality'], axis=1)
            output_data = train_data[['quality']]
            X_train, X_test, y_train, y_test = train_test_split( input_data, output_data, test_size=0.3, random_state=42)
            with mlflow.start_run():
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ration, random_state=42)
                model.fit(X_train, y_train)

                prediction = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, prediction)

                print(f"Elastic model(alpha={alpha}, l1_ratio={l1_ration})")
                print(f"RMSE : {rmse}")
                print(f"MAE : {mae}"),
                print(f"R2 : {r2}")

                '''
                    Single parameters
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("l1_ration", l1_ration)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("r2", r2)
                '''
                '''
                    Multiple parameters
                '''
                mlflow.log_params({
                    "alpha": alpha,
                    "l1_ration": l1_ration
                })

                mlflow.log_metrics({
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2
                })

                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(e)




if __name__ == "__main__":
    mlflow.set_experiment("elasticnet")
    model_v1 = ElasticNetModel()
    # model_v1.train_model(0.5, 0.5)
    model_v1.train_model(0.01, 0.1)
    model_v1.train_model(0.05, 0.05)
    model_v1.train_model(0.20, 0.05)




