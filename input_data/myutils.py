import random
from tabulate import tabulate
import math
import copy
import itertools
from operator import itemgetter


# TODO: FINISH AND COMMENT
def get_mpg_ranking(col):
    '''
    Args:
    Return:
    '''
    ranking = []
    for i in range(len(col)):
        ranking.append(1)
    
    for value in range(len(col)):
        if col[value] >= 13 and col[value] < 14:
            ranking[value] = 1
        elif col[value] == 14:
            ranking[value] = 2
        elif col[value] > 14 and col[value] <= 16:
            ranking[value] = 3
        elif col[value] > 16 and col[value] <= 19:
            ranking[value] = 4
        elif col[value] > 19 and col[value] <= 23:
            ranking[value] = 5
        elif col[value] > 23 and col[value] <= 26:
            ranking[value] = 6
        elif col[value] > 26 and col[value] <= 30:
            ranking[value] = 7
        elif col[value] > 30 and col[value] <= 36:
            ranking[value] = 8
        elif col[value] > 36 and col[value] <= 44:
            ranking[value] = 9
        elif col[value] >= 45:
            ranking[value] = 10
    return ranking

def group_by(X,y):
    '''
    Args:
    Return:
    '''
    labels = []
    counts = []
    for label in y:
        if label not in labels:
            labels.append(label)
            counts.append(1)
        elif label in labels:
            index = labels.index(label)
            counts[index] += 1

    ran = len(y)//len(labels)
    grouped = []
    labelss = []

    for x in range(len(y)):
        if y[x] not in labelss:
            labelss.append(x)
            for i in range(len(labels)):
                for lab in range(len(y)):
                    if y[lab] == labels[i]:
                        grouped.append(lab)
        break
    print(grouped)

    list_of_lists = []
    index = 0
    for row in range(ran):
        inner_list = []
        for col in range(len(labels)):
            if index != len(grouped):
                inner_list.append(grouped[index])
                index = index + 1
        list_of_lists.append(inner_list)
    print(list_of_lists)
            
    return list_of_lists

def get_unique(vals):
    '''Given a list, will return a new list with all the duplicates removed.
    Args:
        vals (list): a list with duplicates
    Returns:
        unique (list): the list without the duplicates
    '''
    unique = []
    for x in vals:
        if x not in unique:
            unique.append(x)

    return unique

def get_priors(y_train):
    '''A prior is calculated by taking the amount of every unique variable from 
    y_train, and finds the amount of instances within the list.
    Args:
        y_train(list of obj): The target y values (parallel to X_train). 
        The shape of y_train is n_samples
    Returns:
        priors(Dictionary): The posterior probabilities computed for each
        attribute value/label pair in the training set.
    '''
    unique = get_unique(y_train) # will return a list of all the unique values in y_train
    priors = {}

    for val in unique:
        priors[val] = 0 # setting the total count of every value in y_train to 0

    # counts up the total of every value
    for val in y_train:
        for u in unique:
            if val == u:
                priors[u] += 1 

    for u in unique:
        priors[u] /= len(y_train)

    return priors

def get_column(table, i):
    '''Identifies a column in a given table and column identifier.
    Args:
        table (list of lists): table used like a database
        i (int): column identifier in the list
    Returns:
        column (list): the specified column from i
    '''
    column = []
    for row in table:
        column.append(row[i])
    
    return column

def get_posteriors(X_train, y_train, priors):
    '''Posteriors are calculated by taking the instances from y_train and X_train,
    and using them in a predict with the priors that we calculated earlier.
    Args:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(Dictionary): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Return:
        posteriors(Dictionary): The posterior probabilities computed for each
        attribute value/label pair in the training set.
    '''
    posteriors = {} 
    cols = []
    attrs = []
    
    # creates the columns in X_train 
    for i in range(len(X_train[0])):
        cols.append(get_column(X_train, i))

    # finds all the unique values in X_train
    for row in cols:
        attrs.append(get_unique(row))

    # goes through prior values and updates the posteriors
    for val, _ in priors.items():
        posteriors[val] = {}
        for i in range(len(attrs)):
            posteriors[val][i] = {}

    # goes through X_train and calculates the posteriors based on the data given
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            class_label = y_train[i]
            attr_val = X_train[i][j]
            if attr_val in posteriors[class_label][j]:
                posteriors[class_label][j][attr_val] = ((posteriors[class_label][j][attr_val] * (priors[class_label] * len(y_train))) + 1) / (priors[class_label] * len(y_train))
            else:
                posteriors[class_label][j][attr_val] = 1/(priors[class_label] * len(y_train))

    return posteriors

def get_prediction_index(vals):
    '''Gets the predicted greatest value in the list.
    Args:
        vals (list): a column or other list form 
    Returns:
        max_index (int): max value within the list
    '''
    max_index = 0
    for i in range(len(vals)):
        if vals[i] > vals[max_index]:
            max_index = i
    
    return max_index

def convert_weight(weight):
    '''Takes in a weight and returns it in a list.
    Args:
        weight (depends):
    Returns:
        res (list): list of all the weight
    '''
    res = []
    for val in weight:
        res.append(get_weight(val))
    return res

def get_weight(val):
    '''Converts the weight to a value of 1-5
    Args:
        val (weight): the weight that will be filled into one of the 5 categories.
    Returns:
        curr (int): a bucket for each of the 5 categories needed for weight.
    '''
    if val < 2000:
        curr = 1
    elif val < 2500:
        curr = 2
    elif val < 3000:
        curr = 3
    elif val < 3500:
        curr = 4
    else:
        curr = 5

    return curr

def get_rating(mpg):
    '''Converts the rating from the miles per gallon.
    Args:
        mpg (int): value of miles per gallon
    Returns:
        int: value corresponding to the mpg
    '''
    if mpg < 14:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    return 10

def convert_to_rating(mpg_list):
    '''Uses the get_rating function to apply it to the whole mpg_list
    Args:
        mpg_list (list): list of the mpg's
    Returns:
        mpg_list (list): arg's list but with the corresponding ratings
    '''
    for i in range(len(mpg_list)):
        mpg_list[i] = get_rating(mpg_list[i])
    return mpg_list

def get_rand_rows(table, num_rows):
    '''Chooses random rows from a given table.
    Args:
        table (list of lists): a given table that will be gone through random rows
        num_rows (int): the amount of rows that will be chosen at random 
    Returns:
        rand_rows (list): the list of rows chosen from table at random
    '''
    rand_rows = []
    for i in range(num_rows):
        rand_rows.append(table.data[random.randint(0, len(table.data)) - 1])
    return rand_rows

def print_pred_actual(rows, actual, predicted):
    '''Prints out the predicted and actual values
    Args:
        rows (list of lists): the list of rows that are used to be tested
        actual (list): what our functions calculate
        predicted (list): what the output should be
    '''
    for i in range(len(rows)):
        print('instance:', rows[i])
        print('class:', predicted[i], 'actual:', actual[i])

def get_accuracy(actual, predicted):
    '''computes the accuracy 
    Args:
        actual (depends): data calculated by hand
        predicted (depends): data calculated from functions
    
    Returns:
        accuracy: the "correctness" of our functions
    '''
    predicted_correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            predicted_correct += 1

    return predicted_correct/len(actual)

def get_from_folds(X_vals, y_vals, train_folds, test_folds):
    '''Gets the train and test folds from the X and y trains
    Args:
        X_vals (list):
        y_vals (list):
        train_folds (list):
        test_folds (list):

    Returns:
        X_train (list): 
        y_train (list):
        X_test (list):
        y_test (list):
    '''
    X_train = []
    y_train = []

    for row in train_folds:
        for i in row:
            X_train.append(X_vals[i])
            y_train.append(y_vals[i])

    X_test = []
    y_test = []

    for row in test_folds:
        for i in row:
            X_test.append(X_vals[i])
            y_test.append(y_vals[i])

    return X_train, y_train, X_test, y_test

# TODO: FINISH AND COMMENT
def add_stats(matrix):
    '''
    Args:
    Returns:
    '''
    del matrix[0]
    for i, row in enumerate(matrix):
        row[0] = i + 1
        row.append(sum(row))
        row.append(round(row[i + 1] / row[-1] * 100, 2))

def print_tabulate(table, headers):
    '''Prints table

    Args:
        table (list of lists): specified data to be printed
        headers (list): labels of each column

    Returns: a table
    '''
    print(tabulate(table, headers, tablefmt="rst"))

def get_values_from_folds(x_vals, y_vals, train_folds, test_folds):
    """Pull the values from the indexed folds from the value lists
    
    Args:
        x_vals - x data set with values
        y_vals - parallel list that is the y data set with values
        train_folds - the determined training folds to pull values from x and y using indices
        test_folds - the determined testing folds to pull values from x and y using indices
    
    Returns:
    """
    # Init lists
    x_train = []
    y_train = []

    # Iterate through training folds
    for row in train_folds:
        for i in row:
            # Append values to lists
            x_train.append(x_vals[i])
            y_train.append(y_vals[i])

    # Init lists
    x_test = []
    y_test = []

    # Iterate through testing folds
    for row in test_folds:
        for i in row:
            # Append values to lists
            x_test.append(x_vals[i])
            y_test.append(y_vals[i])

    return x_train, y_train, x_test, y_test

def split_x_y_train(data):
    """
    Function to split the data into x and y with y being the last column in a table

    Args:
        data: the table to split

    Returns:
        the x_train data list[:-1]
        the y_train data list[-1]
    """
    x_train = []
    y_train = []
    for row in data:
        x_train.append(row[:-1])
        y_train.append(row[-1])
    
    return x_train, y_train

def all_same_class(instances):
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label


def partition_instances(instances, att_index, att_domains):
    '''Partition the list of instances 

    Args:
        instances:
        att_index:
        att_domains:

    Returns:
        partitions:
    '''
    partitions = {}
    attribute_domain = att_domains["att" + str(att_index)]

    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[att_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions


def get_frequencies(col):
    """Gets the frequency and count of a column by name

    Args:
        MyPyTable(MyPyTable): self of MyPyTable
        col_name(str): name of the column

    Returns:
        values, counts (string, int): name of value and its frequency"""

    values = []
    counts = []

    for value in col:
        if value not in values:
            # haven't seen this value before
            values.append(value)
            counts.append(1)
        elif value in values:
                index = values.index(value)
                counts[index] += 1

    return counts, len(col)


def entropy(y):
    """Returns entropy for a distribution of classes

    Args:
        y(list): distribution of classes

    Returns:
        entropy(float): entropy"""
    
    frequencies, length = get_frequencies(y)
    total_arr = []
    
    #calculate ratios
    for col in frequencies:
        total_arr.append(col/length)

    entropy = 0
    #calculate entropies
    for ratio in total_arr:
        log_res = -ratio * math.log2(ratio)
        entropy += log_res

    return entropy


def select_attribute(instances, att_indexes, class_index):
    '''
    Args:
    Returns:
    '''
    y_train = [instance[class_index] for instance in instances]
    info = {}
    first_entropy = entropy(y_train)

    for col in att_indexes:
        att_entropies = {}
        instances = sorted(instances, key=itemgetter(col))

        #reference for using intertools instead of pandas: https://www.geeksforgeeks.org/itertools-groupby-in-python/
        key_func = lambda col: itemgetter(col)
        for key, group in itertools.groupby(instances, key=key_func(col)):
            y = [x[-1] for x in group]
            entropy_val = entropy(y)
            att_entropies[key] = (len(y), entropy_val)

        #get weighted entropy
        second_entropy = 0
        for key, entropies in att_entropies.items():
            second_entropy += (entropies[0] / len(y_train)) * entropies[1]
        
        weighted_entropy = first_entropy - second_entropy
        info[col] = weighted_entropy
    
    #find split_attribute
    split_attribute = sorted(info.items(), key=itemgetter(int(-1)), reverse=True)[0][0]
    for col in att_indexes:
        if col == split_attribute:
            split_attribute = col

    return split_attribute


def check_all_same_class(instances, class_index):
    '''
    Args:
        instances:
        class_index:

    Returns:
        bool:
    '''
    first_label = instances[0][class_index]
    for instance in instances:
        if instance[class_index] != first_label:
            return False 
    return True


def compute_partition_stats(instances, class_index):
    """Computes partition statistics for cases 2 and 3
        Args:
            instances(list): current partition
            class_index(list): attribute used as the class label
        Returns:
            array (list): The correct values to add to tree
        """
    statistic = {}

    for instance in instances:
        if instance[class_index] in statistic:
            statistic[instance[class_index]] += 1
        else:
            statistic[instance[class_index]] = 1

    array = []

    for key in statistic:
        array.append([key, statistic[key]])
    
    return array


# TODO: FIX AND COMMENT
def tdidt(instances, att_indexes, att_domains, class_index):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    split_attribute = select_attribute(instances, att_indexes, class_index)
    att_indexes.remove(split_attribute)
    # cannot split on the same attribute twice in a branch
    tree = ["Attribute", "att" + str(split_attribute)]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(instances, split_attribute, att_domains)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and check_all_same_class(partition, class_index):
            leaf = ["Leaf", partition[0][class_index], len(partition), len(instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(att_indexes) == 0:
            labels = [instance[class_index] for instance in partition]
            majority_vote = max(set(labels), key=labels.count)
            leaf = ["Leaf", majority_vote, len(partition), len(instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            labels = [instance[class_index] for instance in instances]
            majority_vote = max(set(labels), key=labels.count)
            tree = ["Leaf", majority_vote, labels.count(majority_vote), len(instances)]
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, att_indexes.copy(), att_domains, class_index)
            values_subtree.append(subtree)
            tree.append(values_subtree)

    return tree


# TODO: FIX AND COMMENT
def tdidt_predict(header, tree, instance):
    '''Helper function for tdidt predict

    '''
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]

        # finding which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label


def tdidt_print_rules(tree, rule, class_name, default_header, attribute_names):
    info_type = tree[0]

    if info_type == "Attribute":
        if rule != "":
            rule += " AND "

        if attribute_names is None: 
            rule += tree[1]
        else:
            index = default_header.index(tree[1])
            rule += attribute_names[index]
            
        for i in range(2, len(tree)):
            value_list = tree[i]
            rule2 = str(rule) + " = " + str(value_list[1])
            tdidt_print_rules(value_list[2], rule2, class_name, default_header, attribute_names)

    else: # "Leaf"
        print(rule, "THEN", class_name, "=", tree[1])

