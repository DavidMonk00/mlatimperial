import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from utils.pipeline import Pipeline


DATA_PATH = "data/week-one/"

models = {
    "Ridge": Ridge(),
    # "LinearRegression": LinearRegression(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "BayesianRidge": BayesianRidge(),
    # "ARDRegression": ARDRegression(),
    "SVR": SVR(),
    "NuSVR": NuSVR(),
    # "KernelRidge": KernelRidge(),
    # "GaussianProcessRegressor": GaussianProcessRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    # "MLPRegressor": MLPRegressor(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor()
}


class HousingPipeline(Pipeline):
    def loadData(self, train_filename, test_filename, macro_filename):
        self.data = pd.read_csv(
            os.path.join(DATA_PATH, train_filename),
            parse_dates=['timestamp'])
        self.test = pd.read_csv(
            os.path.join(DATA_PATH, test_filename),
            parse_dates=['timestamp'])
        self.macro = pd.read_csv(
            os.path.join(DATA_PATH, macro_filename),
            parse_dates=['timestamp'])

    def preprocess(self):
        self.data.drop(['id', 'price_doc'], axis=1, inplace=True)
        self.X = pd.merge_ordered(
            self.data.copy(), self.macro.copy(), on='timestamp', how='left')
        self.X.fillna(0, inplace=True)

        # Take only numeric data for now
        self.X = self.X.select_dtypes(exclude=['object'])
        self.X.drop(columns=["timestamp"], inplace=True)

        # Scale data
        self.scaler = StandardScaler()
        self.features = self.X.copy().columns
        self.X = self.scaler.fit_transform(self.X)

    def train(self, model):
        super().train(model)

        training_ind, validation_ind = train_test_split(
            range(len(self.X)),
            train_size=0.10
        )
        self.model.fit(self.X[training_ind], np.log1p(self.Y[training_ind]))
        return mean_squared_error(
            self.model.predict(
                self.X[validation_ind]), np.log1p(self.Y[validation_ind]),
            squared=False
        )

    def predict(self):
        pred_ids = self.test['id']
        self.test.drop(['id'], axis=1, inplace=True)
        self.X_predict = pd.merge_ordered(
            self.test, self.macro, on='timestamp', how='left')
        self.X_predict.fillna(0, inplace=True)
        self.X_predict = self.X_predict[self.features]
        predictions = np.expm1(
            self.model.predict(self.scaler.transform(self.X_predict)))
        predictions = pd.DataFrame(predictions, columns=["price_doc"])
        predictions = pd.concat([pred_ids, predictions], axis=1)
        return predictions


def uploadToKaggle(predictions):
    predictions.to_csv(os.path.join(DATA_PATH, "predictions.csv"), index=False)


def main():
    hp = HousingPipeline()
    hp.loadData("X_train.csv", "X_test.csv", "macro.csv")

    print(hp.data.shape, hp.test.shape, hp.macro.shape)

    hp.setTarget("price_doc")
    hp.preprocess()
    results = {}
    for model in models:
        print(model)
        results[model] = hp.train(models[model])
    # uploadToKaggle(hp.predict())
    # print(models)
    print(pd.DataFrame(
        list(results.items()), columns=["model", "error"],
        index=np.arange(len(results))))


if __name__ == '__main__':
    main()
