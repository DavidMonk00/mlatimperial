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
    # "ridge": {
    #     'model': Ridge(),
    #     'param_grid': {
    #         'ridge__alpha': np.logspace(-3, 4, 5)
    #     }
    # },
    # # "LinearRegression": LinearRegression(),
    # "lasso": {
    #     'model': Lasso(),
    #     'param_grid': {
    #         'lasso__alpha': np.logspace(-3, 4, 5)
    #     }
    # },
    # "elasticnet": {
    #     'model': ElasticNet(),
    #     'param_grid': {
    #         'elasticnet__alpha': np.logspace(-3, 4, 5)
    #     }
    # },
    # "BayesianRidge": BayesianRidge(),
    # # "ARDRegression": ARDRegression(),
    "svr": {
        'model': SVR(),
        'param_grid': {
            'svr__C': np.logspace(-3, 3, 10)
        }
    },
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
        self.setTarget("price_doc")
        self.data.drop(['id', 'price_doc'], axis=1, inplace=True)
        self.X = pd.merge_ordered(
            self.data.copy(), self.macro.copy(), on='timestamp', how='left')
        self.X.fillna(0, inplace=True)

        # Take only numeric data for now
        self.X = self.X.select_dtypes(exclude=['object'])
        self.X.drop(columns=["timestamp"], inplace=True)

    def construct(self, *args):
        self.pipeline = make_pipeline(*args)
        print(self.pipeline)

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

    def search(self, param_grid):
        gscv = GridSearchCV(
            self.pipeline, param_grid, n_jobs=-1,
            scoring='neg_root_mean_squared_error', verbose=1, cv=5,
            refit='best_index_'
        )
        gscv.fit(self.X, np.log1p(self.Y))
        print(pd.DataFrame(gscv.cv_results_))
        print(gscv.best_estimator_)
        print(gscv.best_score_)

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

    hp.preprocess()
    for model in models:
        hp.construct(StandardScaler(), PCA(), models[model]['model'])
        param_grid = {
            # "pca__n_components": np.logspace(0, 2, 5, dtype=np.int64)
        }
        param_grid = {**param_grid, **models[model]['param_grid']}
        hp.search(param_grid)
    # print(gscv)
    # results = {}
    # for model in models:
    #     print(model)
    #     results[model] = hp.train(models[model])
    # # uploadToKaggle(hp.predict())
    # # print(models)
    # print(pd.DataFrame(
    #     list(results.items()), columns=["model", "error"],
    #     index=np.arange(len(results))))


if __name__ == '__main__':
    main()
