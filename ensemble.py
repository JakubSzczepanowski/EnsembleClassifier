
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
import pandas as pd
import math

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_estimator: BaseEstimator, n_estimators: int = 10, rand_features: bool = True):
        self._n_estimators: int = n_estimators
        self._estimators: list[BaseEstimator] = []
        self.base_estimator: BaseEstimator = base_estimator
        self.rand_features: bool = rand_features
        self._feature_indexes: list = []

    def fit(self, X: pd.DataFrame, y: np.array):
        X = X.reset_index(drop=True)
        n = len(X)
        for _ in range(self._n_estimators):
            #bootstrapowe losowanie ze zwracaniem
            bootstrap_samples = np.random.randint(n, size=n)
            if self.rand_features:
                features_indexes = np.random.choice(X.columns, size=math.floor(math.sqrt(len(X.columns))), replace=False)
                X_copy = X[features_indexes].copy()
                self._feature_indexes.append(features_indexes)
            new_estimator = clone(self.base_estimator).fit(X_copy.iloc[bootstrap_samples], y[bootstrap_samples])
            self._estimators.append(new_estimator)
        return self

    def predict(self, X: pd.DataFrame) -> np.array:
        results = [None]*len(X.index)

        if callable(getattr(self.base_estimator, 'predict', None)):
            self._hard_voting(results, X)
        else:
            raise NameError('No predict function on base estimator')
        
        return np.array(results)
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        results = [None]*len(X.index)

        if callable(getattr(self.base_estimator, 'predict_proba', None)):
            self._soft_voting(results, X)
        else:
            raise NameError('No predict_proba function on base estimator')
        
        return np.array(results)
    
    def _hard_voting(self, results: list, X: pd.DataFrame):
        #iteracja po każdym wierszu
        for index, row_index in enumerate(X.index):
            row = X.loc[row_index]
            counter = {}
            max_votes = ('', 0)
            #predykcja dla każdego estymatora i głosowanie
            for estimator_index, estimator in enumerate(self._estimators):
                temp_row = row.copy()
                if self.rand_features:
                    temp_row = temp_row[self._feature_indexes[estimator_index]]
                row_array = np.array(temp_row).reshape(1,-1)
                prediction = estimator.predict(row_array)[0]
                if prediction in counter:
                    counter[prediction] += 1
                else:
                    counter[prediction] = 1
                if counter[prediction] > max_votes[1]:
                    max_votes = (prediction, counter[prediction])
            results[index] = max_votes[0]

    def _soft_voting(self, results: list, X: pd.DataFrame):
        for index, row_index in enumerate(X.index):
            row = X.loc[row_index]
            sum = 0
            for estimator_index, estimator in enumerate(self._estimators):
                temp_row = row.copy()
                if self.rand_features:
                    temp_row = temp_row[self._feature_indexes[estimator_index]]
                row_array = np.array(temp_row).reshape(1,-1)
                prediction = estimator.predict_proba(row_array).flatten()
                sum += prediction
            results[index] = sum/self._n_estimators