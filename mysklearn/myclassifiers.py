"""
Programmer: Alex Giacobbi and Joseph Torii
Class: CPSC 322-02, Spring 2021
Semester Project
22 April 2021

Description: This file includes all of the classifiers that we have done throughout 
the PA's and the new MyRandomForestClassifier. 
"""

import mysklearn.myutils as myutils
import numpy as np
import operator
import copy

from collections import Counter

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        mean_x = np.mean(X_train)
        mean_y = np.mean(y_train)
        
        self.slope = sum([(X_train[i] - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train))]) / sum([(X_train[i] - mean_x) ** 2 for i in range(len(X_train))])
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        return [self.slope * val + self.intercept for val in X_test]

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier tcheck_all_same_classy_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for example in X_test:
            test_distances = []
            for i in range(len(self.X_train)):
                dist = 0
                continuous_vals = 0
                for j in range(len(example)):
                    if isinstance(example[j], float) or isinstance(example[j], int):
                        continuous_vals += ((example[j] - self.X_train[i][j]) ** 2)
                    else:
                        dist += 1
                continuous_vals = np.sqrt(continuous_vals)
                dist += continuous_vals
                test_distances.append((i, dist))
            test_distances_sorted = sorted(test_distances, key=operator.itemgetter(1))
            distances.append([test_distances_sorted[i][1] for i in range(self.n_neighbors)])
            neighbor_indices.append([test_distances_sorted[i][0] for i in range(self.n_neighbors)])

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, neighbors = self.kneighbors(X_test)

        for example in neighbors:
            labels = [self.y_train[neighbor] for neighbor in example]
            y_predicted.append(max(set(labels), key=labels.count))

        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(dict of str : float): The prior probabilities computed for each
            label in the training set.
        posteriors(list of float): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.priors = {}
        self.posteriors = {}

        for val in y_train:
            if val in self.priors:
                self.priors[val] += 1
            else:
                self.priors[val] = 1

        for key in self.priors:
            self.priors[key] /= len(y_train)

        num_attributes = len(X_train[0])

        for key in self.priors:
            post = []
            for i in range(num_attributes):
                vals = []
                all_vals = set()
                instance_dict = {}
                for j in range(len(X_train)):
                    if y_train[j] == key:
                        vals.append(X_train[j][i])
                    all_vals.add(X_train[j][i])
                for val in all_vals:
                    instance_dict[val] = ((vals.count(val)) * 1.0) / (len(vals))
                post.append(instance_dict)
            self.posteriors[key] = post



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for example in X_test:
            probailities = []
            classes = []
            for key in self.priors:
                product = self.priors[key]
                for i in range(len(example)):
                    product *= self.posteriors[key][i][example[i]]
                probailities.append(product)
                classes.append(key)

            y_predicted.append(classes[probailities.index(max(probailities))])

        return y_predicted

class MyZeroRClassifier:
    """Represents a Zero-R classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        prediction(obj): The prediction that this classifier will always select.
    """
    def __init__(self):
        """Initializer for MyZeroRClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.prediction = None

    def fit(self, X_train, y_train):
        """Fits a Zero-R classifier to y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train 
        self.y_train = y_train
        self.prediction = max(set(y_train), key = y_train.count)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.prediction for _ in X_test]

class MyRandomClassifier:
    """Represents a random classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    """
    def __init__(self):
        """Initializer for MyRandomClassifier.

        """
        self.X_train = None 
        self.y_train = None


    def fit(self, X_train, y_train):
        """Fits a Zero-R classifier to y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train 
        self.y_train = y_train


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.y_train[np.random.randint(0, len(self.y_train))] for _ in X_test]

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(X_train)

        # make a dictionary of possible values in the form {attribute: values}
        available_attributes = {}
        for i in range(0, len(X_train[0])):
            att = "att" + str(i)
            available_attributes[att] = []
            for x in X_train:
                if x[i] not in available_attributes[att]:
                    available_attributes[att].append(x[i])

        for i, x in enumerate(y_train):
            X_train[i].append(x)

        self.tree = myutils.tdidt(X_train, [x for x in range(0, len(X_train[0]) - 1)], available_attributes, -1)
        
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        header = ['att' + str(i) for i in range(len(self.X_train[0]))]  # Computing headers

        for x in X_test:
            y_predicted.append(myutils.tdidt_predict(header, self.tree, x))

        return y_predicted


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        default_header = ["att" + str(i) for i in range(0, len(self.X_train))]
        myutils.tdidt_print_rules(self.tree, "", class_name, default_header, attribute_names)


# TODO: FINISH AND COMMENT
class MyRandomForestClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        N (int): The number of trees to be trained
        M (int): The number of classifiers from the set of N trees to keep (based on accuracy)
        F (int): The number of attributes to be randomly sampled from the training set
        learners (list of MyDecisionTreeClassifier): list of weak learners in the ensemble
        accuracies (list of float): accuracies of each of the learners on the validation set
            parallel to the learners list and used to select the M trees
            
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, N, M, F, seed):
        """Initializer for MyRandomForestClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.N = N
        self.M = M
        self.F = F
        self.seed = None
        self.learners = None
        self.accuracies = 0

    def fit(self, X_train, y_train):
        ''' Fits the random forest model to a given training set
        
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train).
                The shape of y_train is n_samples
        '''
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)
        self.learners = []
        self.accuracies = []

        # generate N learners
        for i in range(self.N):
            # create the bootstrap sample with or without seed
            if self.seed is None:
                X_sample, Y_sample = myutils.compute_sample(self.X_train, self.y_train)
            else:
                X_sample, Y_sample = myutils.compute_sample(self.X_train, self.y_train, self.seed)

            # create the validation set
            X_val = [x for x in self.X_train if x not in X_sample]
            y_idxs = [self.X_train.index(x) for x in X_val]
            y_val = [self.y_train[i] for i in range(len(self.y_train)) if i in y_idxs]

            # get only a random subset of attributes for each sample
            values = [x for x in range(len(self.X_train[0]))] 

            if self.seed is None:
                F_attributes = myutils.compute_random_subset(values, self.F)
            else:
                F_attributes = myutils.compute_random_subset(values, self.F, self.seed)

            # get only those attributes from the training set
            for i in range(len(X_sample)):
                X_sample[i] = [X_sample[i][j] for j in range(len(X_sample[i])) if j in F_attributes]

            # get only those attributes from the validation set
            for i in range(len(X_val)):
                X_val[i] = [X_val[i][j] for j in range(len(X_val[i])) if j in F_attributes]

            # build a decision tree from the sample
            tree = MyDecisionTreeClassifier()
            tree.fit(X_sample, Y_sample)
            self.learners.append(tree)

            # test the trees accuracy on the validation set
            y_pred = tree.predict(X_val)

            self.accuracies.append(myutils.compute_accuracy(y_pred, y_val))

        # sort the dists and move the indices to match the sorted list 
        # by combining the two lists into a list of tuples, sorting, and unpacking
        sorted_accuracies, sorted_idxs = (list(x) for x in zip(*sorted(zip(self.accuracies, range(len(self.learners))))))

        # slice the lists to only include the M best learners
        self.learners = [self.learners[i] for i in range(len(self.learners)) if i in sorted_idxs[:self.M]]
        self.accuracies = sorted_accuracies[:self.M]


    def predict(self, X_test):
        ''' Predicts the class labels of a set of test instances
        Args:
            X_test (list of list of obj): The list of test instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        Returns:
            y_predicted (list of labels): labels corresponding to the test set
        '''
        # get predictions from all of the trees
        all_predictions = []
        for tree in self.learners:
            predicted = tree.predict(X_test)
            all_predictions.append(predicted)

        y_predicted = []
        # get the most common prediction for each x value
        for i in range(len(X_test)):
            x_preds = [p[i] for p in all_predictions]

            # get most common prediction (majority vote)
            majority = Counter(x_preds).most_common(1)
            majority_label = list(majority[0])[0] # unpack the object to get the label
            y_predicted.append(majority_label)

        return y_predicted


