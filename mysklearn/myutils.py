from mysklearn.mypytable import MyPyTable
import os
import random
import math
from collections import Counter
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np


def frequency_diagram(frequencies, title, x_label, y_label):
    # plot frequencies
    categories = [key for key in frequencies.keys()]
    categories.sort()
    xs = [i for i in range(len(categories))]
    ys = [frequencies[key] for key in categories]

    plt.figure(figsize=(40, 10))
    plt.bar(xs, ys)
    plt.xticks(xs, categories, rotation='vertical')

    # label diagram
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def pie_chart(values, labels, title):
    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.show()


def histogram(values, title, x_label, y_label):
    plt.figure()
    plt.hist(values)

    # label diagram
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_and_fit(xs, ys, title, x_label, y_label):
    plt.figure()
    plt.scatter(xs, ys, linewidths=0.5, alpha=0.5)

    m, b = compute_slope_intercept(xs, ys)
    plt.plot([min(xs), max(xs)], [m * min(xs) + b, m * max(xs) + b], c="r", lw=5)

    # label diagram
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def compute_slope_intercept(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return m, b 


def box_plot(distributions, labels, title, x_label, y_label):
    plt.figure()
    plt.boxplot(distributions)

    plt.xticks(list(range(1, len(labels) + 1)), labels, rotation='vertical')

    # label diagram
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def load_data(filename):
    data_path = os.path.join("input_data", filename)
    table = MyPyTable().load_from_file(data_path)
    
    return table


def convert_mpg_rating(mpg_list):
    mpg_ratings = []
    for value in mpg_list:
        if value >= 45:
            mpg_ratings.append(10)
        elif value >= 37:
            mpg_ratings.append(9)
        elif value >= 31:
            mpg_ratings.append(8)
        elif value >= 26:
            mpg_ratings.append(7)
        elif value >= 24:
            mpg_ratings.append(6)
        elif value >= 20:
            mpg_ratings.append(5)
        elif value >= 17:
            mpg_ratings.append(4)
        elif value >= 15:
            mpg_ratings.append(3)
        elif value >= 14:
            mpg_ratings.append(2)
        else:
            mpg_ratings.append(1)

    return mpg_ratings


def normalize_series(data):
    max_x = max(data)
    min_x = min(data)

    return [(x - min_x) / ((max_x - min_x) * 1.0) for x in data]


def evaluate_classifier(y_predicted, y_actual, classes):
    # classes = list(set(y_actual))
    falses = [0 for _ in classes]
    trues = [0 for _ in classes]

    for i in range(len(y_predicted)):
        if y_predicted[i] == y_actual[i]:
            trues[classes.index(y_actual[i])] += 1
        else:
            falses[classes.index(y_predicted[i])] += 1

    accuracy = (sum(trues)) / (sum(trues) + sum(falses))
    error_rate = 1 - accuracy

    return accuracy, error_rate


def convert_weight_rating(weight):
    if weight >= 3500:
        return 5
    elif weight >= 3000:
        return 4
    elif weight >= 2500:
        return 3
    elif weight >= 2000:
        return 2
    else:
        return 1


def check_all_same_class(instances, class_index):
    first_label = instances[0][class_index]
    for instance in instances:
        if instance[class_index] != first_label:
            return False 
    return True


def entropy(y):    
    frequencies = Counter(y)
    total_arr = []
    
    for val in frequencies:
        total_arr.append(frequencies[val] / len(y))

    entropy = 0
    for ratio in total_arr:
        log_res = -ratio * math.log2(ratio)
        entropy += log_res

    return entropy


def select_attribute(instances, att_indexes, class_index):
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


def partition_instances(instances, att_index, att_domains):
    partitions = {}
    attribute_domain = att_domains["att" + str(att_index)]

    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[att_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions
    

# TODO: FIX AND COMMENT
def tdidt(instances, att_indexes, att_domains, class_index):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    split_attribute = select_attribute(instances, att_indexes, class_index)
    att_indexes.remove(split_attribute)
    # cannot split on the same attribute twice in a branch
    # recall: python is pass by object reference!!
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
            maj_vote = max(set(labels), key=labels.count)
            leaf = ["Leaf", maj_vote, len(partition), len(instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            labels = [instance[class_index] for instance in instances]
            maj_vote = max(set(labels), key=labels.count)
            tree = ["Leaf", maj_vote, labels.count(maj_vote), len(instances)]
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, att_indexes.copy(), att_domains, class_index)
            values_subtree.append(subtree)
            tree.append(values_subtree)

    return tree


def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]

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


def load_data(filename):
    data_path = os.path.join("input_data", filename)
    table = MyPyTable().load_from_file(data_path)
    
    return table


def count_column_frequencies(table, column_name):
    column_data = table.get_column(column_name, include_missing_values=False)
    return count_frequency(column_data)


def count_frequency(data):
    freq = {}

    for value in data:
        value = str(value)
        if value in freq.keys():
            freq[value] += 1
        else:
            freq[value] = 1

    return freq

def get_column_sum(table, column_name):
    table.convert_to_numeric()
    column = table.get_column(column_name, include_missing_values=False)
    return sum(column)


def convert_to_bins(mpg_data, num_bins):
    lower_bound = min(mpg_data)
    upper_bound = max(mpg_data)
    bin_size = (upper_bound - lower_bound) / num_bins
    mpg_ratings = []

    for value in mpg_data:
        for i in range(1, num_bins + 1):
            if value >= (upper_bound - (i * bin_size)):
                mpg_ratings.append(num_bins - i + 1)
                break

    return mpg_ratings

def strip_percent(data):
    clean_data = []
    for val in data:
        val = val[:-1]
        try:
            val = float(val)
            clean_data.append(val)
        except:
            pass

    return clean_data


def parse_multiple_values(column):
    flat_list = []
    for row in column:
        vals = row.split(",")
        for val in vals:
            flat_list.append(val)

    return flat_list


def parse_genres(data):
    genres = set()
    for row in data:
        row_gen = row.split(",")
        for genre in row_gen:
            genres.add(genre)

    return genres


def group_by(table, group_by_col_name):
    col = table.get_column(group_by_col_name)
    col_index = table.column_names.index(group_by_col_name)
    col = parse_genres(col)
    
    # we need the unique values for our group by column
    group_names = sorted(list(col)) # e.g. 74, 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate
    # subtable based on its group_by_col_name value
    for row in table.data:
        group_by_value = row[col_index]
        # which subtable to put this row in?
        group_indexes = []
        for group in group_by_value.split(","):
            group_indexes.append(group_names.index(group))

        for group_index in group_indexes:
            group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables