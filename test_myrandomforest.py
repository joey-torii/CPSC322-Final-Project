import numpy as np
import itertools
import mysklearn.myutils as myutils

from mysklearn.myclassifiers import MyRandomForestClassifier

def compare_rules(actual, expected):
    assert actual['lhs'] == expected['lhs']
    assert actual['rhs'] == expected['rhs']
    assert np.isclose(actual['confidence'], expected['confidence'])
    assert np.isclose(actual['support'], expected['support'])
    assert np.isclose(actual['lift'], expected['lift'])



def random_forest_classifier_fit():
     # interview dataset
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    table = [
        ["level=Senior", "lang=Java", "tweets=no", "phd=no", "interviewed_well=False"],
        ["level=Senior", "lang=Java", "tweets=no", "phd=yes", "interviewed_well=False"],
        ["level=Mid", "lang=Python", "tweets=no", "phd=no", "interviewed_well=True"],
        ["level=Junior", "lang=Python", "tweets=no", "phd=no", "interviewed_well=True"],
        ["level=Junior", "lang=R", "tweets=yes", "phd=no", "interviewed_well=True"],
        ["level=Junior", "lang=R", "tweets=yes", "phd=yes", "interviewed_well=False"],
        ["level=Mid", "lang=R", "tweets=yes", "phd=yes", "interviewed_well=True"],
        ["level=Senior", "lang=Python", "tweets=no", "phd=no", "interviewed_well=False"],
        ["level=Senior", "lang=R", "tweets=yes", "phd=no", "interviewed_well=True"],
        ["level=Junior", "lang=Python", "tweets=yes", "phd=no", "interviewed_well=True"],
        ["level=Senior", "lang=Python", "tweets=yes", "phd=yes", "interviewed_well=True"],
        ["level=Mid", "lang=Python", "tweets=no", "phd=yes", "interviewed_well=True"],
        ["level=Mid", "lang=Java", "tweets=yes", "phd=no", "interviewed_well=True"],
        ["level=Junior", "lang=Python", "tweets=no", "phd=yes", "interviewed_well=False"]
    ]
    rules = [
        {'lhs': ['interviewed_well=False'], 'rhs': ['tweets=no'], 'confidence': 0.8, 'support': 0.2857142857142857, 'lift': 1.6},
        {'lhs': ['level=Mid'], 'rhs': ['interviewed_well=True'], 'confidence': 1.0, 'support': 0.2857142857142857, 'lift': 1.55555555556},
        {'lhs': ['tweets=yes'], 'rhs': ['interviewed_well=True'], 'confidence': 0.8571428571428571, 'support': 0.42857142857142855, 'lift': 1.33333333333},
        {'lhs': ['lang=R'], 'rhs': ['tweets=yes'], 'confidence': 1.0, 'support': 0.2857142857142857, 'lift': 2.0},
        {'lhs': ['phd=no', 'tweets=yes'], 'rhs': ['interviewed_well=True'], 'confidence': 1.0, 'support': 0.2857142857142857, 'lift': 1.55555555556},
    ]

    rfc = MyRandomForestClassifier(20, 7, 2, None)

    for i in range(len(rules)):
        compare_rules(rules[i], arm.rules[i])

    assert 1 == 1

def random_forest_classifier_predict():
    assert 1 == 1