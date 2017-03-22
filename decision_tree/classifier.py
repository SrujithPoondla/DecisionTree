import pandas as pd
import scipy.io.arff as arff
import numpy as np
import sys

from dt_learn import learn, predict


def preprocess(file):
    loaded_arff = arff.loadarff(open(file, 'rb'))
    (training_data, metadata) = loaded_arff
    features = metadata.names()

    training_data = pd.DataFrame(training_data)
    training_data.loc[training_data["class"] == "negative", "class"] = 0
    training_data.loc[training_data["class"] == "positive", "class"] = 1
    del features[-1]
    return training_data, features, metadata


def accuracy(df, predicted_column):
    matched_values = pd.DataFrame({'predicted': df['class'] == predicted_column})
    corrent_prediction = len(matched_values[matched_values['predicted'] == True])
    return (float(corrent_prediction) / len(df)) * 100


def printMatches(df, predicted):
    def decode_class_as_string(x):
        return "negative" if x == 0 else "positive"

    print("<Predictions for the Test Set Instances>")
    actual_predicted = zip(df['class'], predicted)
    for idx, tuple in enumerate(actual_predicted):
        print("{0}: Actual: {1} Predicted: {2}".format(idx + 1, decode_class_as_string(tuple[0]),
                                                       decode_class_as_string(tuple[1])))

    matched_values = pd.DataFrame({'predicted': df['class'] == predicted})
    corrent_prediction = len(matched_values[matched_values['predicted'] == True])
    print("Number of correctly classified: {0} Total number of test instances: {1}".format(corrent_prediction, len(df)))


def sample(df, percentage):
    if percentage == 100:
        return df
    row_count = int(len(df) * (percentage / float(100)))
    rows = np.random.choice(df.index.values, row_count)
    return df.ix[rows]


def learner_accuracy(df, learner):
    predicted_column = predict(df, learner)
    return accuracy(df, predicted_column)

if __name__ == '__main__':
    df, features, metadata = preprocess(sys.argv[1])
    learner = learn(df, features, metadata, int(sys.argv[3]))
    print(str(learner))

    test_df, _, _ = preprocess(sys.argv[2])
    predicted_column = predict(df, learner)
    predicted_column = predict(test_df, learner)
    printMatches(test_df, predicted_column)


