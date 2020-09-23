import numpy as np
import random
import sklearn
import sklearn.datasets as ds


class CreateDataset:
    """docstring for CreateDataset"""
    def __init__(self, n_train, n_test, n_features, n_redundant, n_classes, neg_class):
        self.n_train = n_train
        self.n_test = n_test
        self.n_features = n_features
        self.n_redundant = n_redundant
        self.n_classes = n_classes
        self.neg_class = neg_class

        
    def classification(self):
        X, y = ds.make_classification(
        n_samples=self.n_train + self.n_test, n_features = self.n_features, n_redundant = self.n_redundant, n_classes = self.n_classes)
        #X= np.column_stack((X,np.ones((X.shape[0],10))))

        sklearn.preprocessing.normalize(X, norm='l2')
        X_train = X[0:self.n_train]
        y_train = y[0:self.n_train]
        X_test = X[self.n_train:]
        y_test = y[self.n_train:]
        y_train[np.where(y_train == 0)] = self.neg_class
        y_test[np.where(y_test == 0)] = self.neg_class
        return X_train, y_train, X_test, y_test

    #random samples and features
    def classification2(self):
        n_samples = random.randint(50,200)
        n_train = int(0.8*n_samples)
        n_test = n_samples-n_train
        n_features = random.randint(2,10)
        X,y = ds.make_classification(n_samples = n_samples, n_features = n_features, n_redundant = self.n_redundant, n_classes = self.n_classes)
        sklearn.preprocessing.normalize(X, norm='l2')
        X_train = X[0:n_train]
        y_train = y[0:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
        y_train[np.where(y_train == 0)] = self.neg_class
        y_test[np.where(y_test == 0)] = self.neg_class
        return X_train, y_train, X_test, y_test
