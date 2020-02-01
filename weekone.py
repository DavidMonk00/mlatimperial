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
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from utils.pipeline import Pipeline


DATA_PATH = "data/week-one/"

models = {
    "ridge": {
        'model': Ridge(),
        'param_grid': {
            'ridge__alpha': np.logspace(-4, 4, 20)
        }
    },
    # "LinearRegression": LinearRegression(),
    "lasso": {
        'model': Lasso(),
        'param_grid': {
            'lasso__alpha': np.logspace(-4, 4, 20)
        }
    },
    "elasticnet": {
        'model': ElasticNet(),
        'param_grid': {
            'elasticnet__alpha': np.logspace(-4, 4, 20)
        }
    },
    # "BayesianRidge": BayesianRidge(),
    # # "ARDRegression": ARDRegression(),
    # "svr": {
    #     'model': SVR(),
    #     'param_grid': {
    #         'svr__C': np.logspace(-3, 3, 10)
    #     }
    # },
    # "NuSVR": NuSVR(),
    # # "KernelRidge": KernelRidge(),
    # # "GaussianProcessRegressor": GaussianProcessRegressor(),
    # "DecisionTreeRegressor": DecisionTreeRegressor(),
    # # "MLPRegressor": MLPRegressor(),
    # "PassiveAggressiveRegressor": PassiveAggressiveRegressor()
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

        def reduce(data):
            correlations = data.corr().abs()
            upper = correlations.where(
                np.triu(np.ones(correlations.shape), k=1).astype(np.bool))
            to_drop = [
                column for column in upper.columns if any(upper[column] > 0.90)
            ]
            return data.drop(columns=to_drop)

        def inpute(data, feature):

            X = data.copy().drop(columns=[feature])
            X = X.select_dtypes(exclude=['object'])
            X = X.fillna(X.median())
            y = data[feature]
            X_train = X[~y.isna()]
            X_test = X[y.isna()]
            y_train = y[~y.isna()]

            model = MLPRegressor()
            model.fit(X_train, y_train)
            print("Feature: %s | Loss = " % feature, model.loss_)
            filled_gaps = model.predict(X_test)
            for i, ind in enumerate(data[feature][data[feature].isna()].index):
                data.at[ind, feature] = filled_gaps[i]
            return data

        self.setTarget("price_doc")
        self.data.drop(['id', 'price_doc'], axis=1, inplace=True)
        # self.X = pd.merge_ordered(
        #     self.data.copy(), self.macro.copy(), on='timestamp', how='left')
        self.X = self.data.copy()
        # self.X.fillna(self.X.median(), inplace=True)

        # Take only numeric data for now
        self.X = self.X.select_dtypes(exclude=['object'])
        self.X.drop(columns=["timestamp"], inplace=True)
        self.X = reduce(self.X)

        for column in self.X.columns[self.X.isna().any() == True]:
            self.X = inpute(self.X, column)

        print("Data shape:", self.X.shape)

    def construct(self, *args):
        self.pipeline = make_pipeline(*args)
        print(self.pipeline)

    def train(self, model=None):
        super().train(model)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        self.pipeline.fit(X_train, np.log1p(y_train))
        return mean_squared_error(
            self.pipeline.predict(X_test),
            y_test,
            squared=False
        )

    def search(self, param_grid, **kwargs):
        gscv = GridSearchCV(
            self.pipeline, param_grid, n_jobs=-1,
            scoring='neg_root_mean_squared_error', verbose=1, cv=5,
            refit='best_index_', **kwargs
        )
        gscv.fit(self.X, np.log1p(self.y))
        print(pd.DataFrame(gscv.cv_results_))
        print(gscv.best_estimator_)
        print(gscv.best_score_)
        self.best_model = gscv.best_estimator_

    def predict(self):
        pred_ids = self.test['id']
        self.test.drop(['id'], axis=1, inplace=True)
        self.X_predict = pd.merge_ordered(
            self.test, self.macro, on='timestamp', how='left')
        self.X_predict.fillna(self.X_predict.median(), inplace=True)
        self.X_predict = self.X_predict[self.X.columns]
        predictions = np.expm1(
            self.best_model.predict(self.X_predict))
        predictions = pd.DataFrame(predictions, columns=["price_doc"])
        predictions = pd.concat([pred_ids, predictions], axis=1)
        return predictions


def uploadToKaggle(predictions):
    predictions.to_csv(os.path.join(DATA_PATH, "predictions.csv"), index=False)


def main():
    hp = HousingPipeline()
    hp.loadData("X_train.csv", "X_test.csv", "macro.csv")

    print(hp.data.shape, hp.test.shape, hp.macro.shape)

    hp.preprocess()
    hp.construct(StandardScaler(), PCA(), ElasticNet())
    hp.search({"elasticnet__alpha": np.logspace(-4, 1, 20)})

    # print(hp.train())
    # for model in models:
    #     hp.construct(StandardScaler(), PCA(), models[model]['model'])
    #     param_grid = {
    #         # "pca__n_components": np.logspace(0, 2, 5, dtype=np.int64)
    #     }
    #     param_grid = {**param_grid, **models[model]['param_grid']}
    #     hp.search(param_grid)
    # print(gscv)
    # results = {}
    # for model in models:
    #     print(model)
    #     results[model] = hp.train(models[model])
    uploadToKaggle(hp.predict())
    # # print(models)
    # print(pd.DataFrame(
    #     list(results.items()), columns=["model", "error"],
    #     index=np.arange(len(results))))


if __name__ == '__main__':
    main()
