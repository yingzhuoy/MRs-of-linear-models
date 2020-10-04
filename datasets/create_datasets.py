import numpy as np
import random
import sklearn
import sklearn.datasets as ds
np.random.seed(1)

class CreateDataset:
    """docstring for CreateDataset"""
    def __init__(self, n_train, n_test, n_redundant, n_classes, neg_class):
        self.n_train = n_train
        self.n_test = n_test
        self.n_redundant = n_redundant
        self.n_classes = n_classes
        self.neg_class = neg_class

        
    def create_dataset(self):
        n_features = random.randint(2,10)
        X, y = ds.make_classification(
            n_samples=self.n_train + self.n_test,n_features = n_features, n_redundant = self.n_redundant, n_classes = self.n_classes)

        sklearn.preprocessing.normalize(X, norm='l2')
        X_train = X[0:self.n_train]
        y_train = y[0:self.n_train]
        X_test = X[self.n_train:]
        y_test = y[self.n_train:]
        y_train[np.where(y_train == 0)] = self.neg_class
        y_test[np.where(y_test == 0)] = self.neg_class
        return X_train, y_train, X_test, y_test, n_features