"""
Programmer: Alex Giacobbi and Joseph Torii
Class: CPSC 322-02, Spring 2021
Semester Project
22 April 2021

Description: This file contains the evaluations that are needed for the Jupyter Notebook visualizations
that will be used within this project.
"""

import mysklearn.myutils as myutils
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       np.random.seed(random_state)
    
    if shuffle: 
        for i in range(len(X)):
            rand_index = np.random.randint(0, len(X))
            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]

    num_instances = len(X)
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    folds = [[] for _ in range(n_splits)]

    for i in range(len(X)):
        folds[i % n_splits].append(i)


    for i in range(len(folds)):
        train_fold = []
        for j in range(len(folds)):
            if i != j:
                train_fold += folds[j]

        X_test_folds.append(folds[i])
        X_train_folds.append(train_fold)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    group_names = sorted(list(set(y))) 
    group_subtables = [[] for _ in group_names]
    folds = [[] for _ in range(n_splits)]
    X_train_folds = []
    X_test_folds = []
    
    # group by class
    for i in range(len(y)):
        group_by_value = y[i]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(i)

    # create stratified folds using groups
    for group in group_subtables:
        for i in range(len(group)):
            folds[i % n_splits].append(group[i])

    # split folds in to train/test
    for i in range(len(folds)):
        train_fold = []
        for j in range(len(folds)):
            if i != j:
                train_fold += folds[j]

        X_test_folds.append(folds[i])
        X_train_folds.append(train_fold)

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in labels] for _ in labels]

    for i in range(len(y_true)):
        row = labels.index(y_true[i])
        col = labels.index(y_pred[i])
        matrix[row][col] += 1

    return matrix