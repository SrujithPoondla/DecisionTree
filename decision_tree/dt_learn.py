from helper import filterDfByLambda
from probabilities import information_gain, binomial_probabilities, class_count_tuple
import re


def numeric_thresholds(df, feature_name):
    new_df = df.sort(columns=feature_name)
    row = new_df[feature_name]
    result = []

    for i in range(len(row) - 1):
        if row.iloc[i] != row.iloc[i + 1]:
            result.append((row.iloc[i] + row.iloc[i + 1]) / 2.0)
    return result


def best_threshold(df, feature_name):
    threshold_tuples = [(threshold, information_gain(df, feature_name, threshold, 'class', numeric=True))
                        for threshold in numeric_thresholds(df, feature_name)]
    return max(threshold_tuples, key=lambda x: x[1])[0]


def isFeatureNumeric(metadata, feature_name):
    return metadata[feature_name][0] == 'numeric'


def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def printTree(root):
    if root.isRoot:
        print(root.feature_name + " - > ")
    else:
        print("({0}, {1}) -> ".format(root.feature_name, root.value))
    for node in root.children:
        print("{0} {1} {2}".format(node.feature_name, node.value, node.probabilities))
    for node in root.children:
        printTree(node)


class Node(object):
    def __init__(self, feature_name, probabilities, value, parent, isRoot=False):
        self.parent = parent
        self.probabilities = probabilities
        self.value = value
        self.isRoot = isRoot
        self.feature_name = feature_name
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self, level=0):
        ret = ""
        for child in self.children:
            ret += child.__str__(0)
        ret = ret[:-1]
        return ret

    def add_child(self, obj):
        self.children.append(obj)

    def max_prob_str(self):
        if self.max_prob() is 0:
            return "negative"
        else:
            return "positive"

    def max_prob(self):
        if self.probabilities[0] > self.probabilities[1]:
            return 0
        elif self.parent is not None and self.probabilities[0] == self.probabilities[1]:
            return self.parent.max_prob();                                                                                                                                                                                                                                                                                                                                                                                                      
        else:
            return 1

    def match(self, df):
        return True

    def predict(self, df):
        if self.match(df):
            if len(self.children) > 0:
                for node in self.children:
                    prediction = node.predict(df)
                    if prediction is not None:
                        return prediction
            return self.max_prob()
        else:
            return None


class NumericNode(Node):
    def __init__(self, feature_name, value, predicate, probabilities, symbol, parent):
        Node.__init__(self, feature_name, probabilities, value, parent)
        self.parent = parent
        self.symbol = symbol
        self.predicate = predicate

    def match(self, df):
        return self.predicate(df[self.feature_name])

    def __str__(self, level=0):
        if is_float(self.value):
            ret = "|\t" * level + "{0} {2} {1:.6f} [{3} {4}]".format(self.feature_name, self.value, self.symbol,
                                                                     self.probabilities[0], self.probabilities[1],
                                                                     self.max_prob_str())
        else:
            ret = "|\t" * level + "{0} {2} {1} [{3} {4}]".format(self.feature_name, self.value, self.symbol,
                                                                 self.probabilities[0], self.probabilities[1],
                                                                 self.max_prob_str()) + "\n"

        if self.is_leaf() is True:
            ret += ": {0}\n".format(self.max_prob_str())
        else:
            ret += "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


class NonNumericNode(Node):
    def __init__(self, feature_name, value, probabilities, parent):
        Node.__init__(self, feature_name, probabilities, value, parent)
        self.parent = parent

    def match(self, df):
        return df[self.feature_name] == self.value

    def __str__(self, level=0):
        if self.is_leaf() is True:
            ret = "|\t" * level + "{0} = {1} [{2} {3}]: {4}".format(self.feature_name, self.value,
                                                                    self.probabilities[0],
                                                                    self.probabilities[1], self.max_prob_str()) + "\n"
        else:
            ret = "|\t" * level + "{0} = {1} [{2} {3}]".format(self.feature_name, self.value, self.probabilities[0],
                                                               self.probabilities[1]) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


def decision_tree(df, features, metadata, depth, depth_limit, instances_count, root):
    if len(df) == 0:
        return None
    if len(df) < instances_count:
        return None
    if len(df) == len(df[df['class'] == 1]) or (len(df) == len(df[df['class'] == 0])):
        return None
    if depth >= depth_limit:
        return None

    best_threshold_dict, info_gain_tuples = info_gain_and_threshold(df, features, metadata)

    best_feature = max(info_gain_tuples, key=lambda x: x[1])[0]
    if isFeatureNumeric(metadata, best_feature):
        greater = lambda x: x > best_threshold_dict[best_feature]
        lesser = lambda x: x <= best_threshold_dict[best_feature]

        leftDF = filterDfByLambda(df, best_feature, lesser)
        rightDF = filterDfByLambda(df, best_feature, greater)
        if len(leftDF) > 0:
            leftNode = NumericNode(best_feature, best_threshold_dict[best_feature], lesser
                                   , class_count_tuple(leftDF['class']), "<=", root)
            root.add_child(leftNode)
            decision_tree(leftDF, features, metadata, depth + 1, depth_limit, instances_count, leftNode)

        if len(rightDF) > 0:
            rightNode = NumericNode(best_feature, best_threshold_dict[best_feature], greater
                                    , class_count_tuple(rightDF['class']), ">", root)
            root.add_child(rightNode)
            decision_tree(rightDF, features, metadata, depth + 1, depth_limit, instances_count, rightNode)

    else:
        for value in metadata[best_feature][1]:
            df_by_feature = df[df[best_feature] == value]
            node = NonNumericNode(best_feature, value, class_count_tuple(df_by_feature['class']), root)
            root.add_child(node)
            decision_tree(df_by_feature, features, metadata, depth + 1, depth_limit, instances_count, node)


def info_gain_and_threshold(df, features, metadata):
    info_gain_tuples = []
    best_threshold_dict = {}
    for feature in features:
        if len(df[feature].unique()) < 2:
            continue
        if isFeatureNumeric(metadata, feature):
            threshold = best_threshold(df, feature)
            best_threshold_dict[feature] = threshold
            feature_info_gain_tuple = (feature, information_gain(df, feature, threshold, 'class', numeric=True))
            info_gain_tuples.append(feature_info_gain_tuple)
        else:
            feature_info_gain_tuple = (feature, information_gain(df, feature, metadata[feature][1], 'class'))
            info_gain_tuples.append(feature_info_gain_tuple)
    return best_threshold_dict, info_gain_tuples


def learn(df, features, metadata, instances_count):
    root = Node("root", binomial_probabilities(df['class']), None, None, isRoot=True)
    decision_tree(df, features, metadata, 0, 100, instances_count, root)
    return root


def predict(df, learner):
    predicted_column = df.apply(lambda x: learner.predict(x), axis=1)
    return predicted_column
