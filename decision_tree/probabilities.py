import math
import numpy as np

from helper import filterDfByLambda


def binomial_probabilities(binomial_values):
    sumOfNos = float(binomial_values.sum())
    prob_postive = sumOfNos / len(binomial_values)
    return 1 - prob_postive, prob_postive


def class_count_tuple(binomial_values):
    sumOfNos = (binomial_values.sum())
    return len(binomial_values) - sumOfNos, sumOfNos


def entropy(probabilities):
    func = np.vectorize(lambda x: 0.0 if (x == 0) else x * math.log(x, 2))
    return -func(probabilities).sum()


def entropy_by_df(df, feature):
    return entropy(np.asarray(binomial_probabilities(df[feature])))


def information_gain(df, feature_name, possible_values, output_feature, numeric=False):
    def conditional_entropy(value):
        if numeric:
            conditional_df = filterDfByLambda(df, feature_name, value)
        else:
            conditional_df = df[df[feature_name] == value]

        if len(conditional_df) == 0:
            return 0
        entropies = entropy_by_df(conditional_df, output_feature)
        return (float(len(conditional_df)) / len(df)) * entropies

    if numeric:
        threshold = possible_values
        possible_values = [(lambda x: x <= threshold), (lambda x: x > threshold)]

    con_entropy = sum([conditional_entropy(value) for value in possible_values])
    entro = entropy_by_df(df, output_feature)
    return entro - con_entropy
