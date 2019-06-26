from functools import wraps
from time import time
import numpy as np
import pickle
import gzip
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"{'➖'*20}\nStart running {f.__name__} function...")
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f"Elapsed time: {end-start}\n{'➖'*20}")
        return result
    return wrapper

@timing
def read_mnist(mnist_file):
    """
    Reads MNIST data.

    Parameters
    ----------
    mnist_file : string
        The name of the MNIST file (e.g., 'mnist.pkl.gz').

    Returns
    -------
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) : tuple
        train_X : numpy array, shape (N=50000, d=784)
            Input vectors of the training set.
        train_Y: numpy array, shape (N=50000)
            Outputs of the training set.
        val_X : numpy array, shape (N=10000, d=784)
            Input vectors of the validation set.
        val_Y: numpy array, shape (N=10000)
            Outputs of the validation set.
        test_X : numpy array, shape (N=10000, d=784)
            Input vectors of the test set.
        test_Y: numpy array, shape (N=10000)
            Outputs of the test set.
    """
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    train_X, train_Y = train_data
    val_X, val_Y = val_data
    test_X, test_Y = test_data

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def pca_fit(X, infomation_presever=.95):
    """TODO: Docstring for pca_fit.
    :returns: TODO

    """
    obj = PCA(infomation_presever)
    obj.fit(X)
    return obj


@timing
def validate_model(param, X_train, y_train, X_val, y_val, N_train=10_000, N_val=10_000, print_score=True):
    """Validate SVM model"""
    print(f"Evaluating model:\n----{param}")
    X_train = X_train[:N_train]
    y_train = y_train[:N_train]
    X_val= X_val[:N_val]
    y_val = y_val[:N_val]
    clf = SVC(**param)
    clf.fit(X_train, y_train)
    print("Caculating score...")
    score = clf.score(X_val, y_val)
    if print_score:
        print("In sample score:", clf.score(X_train, y_train))
        print("Validation score:", score)
    return score, clf
