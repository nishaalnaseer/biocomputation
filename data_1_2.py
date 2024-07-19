import random
import numpy as np
from icecream import ic
import gymnasium as gym
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neural_network import MLPClassifier


def main(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines.pop(0)

        for line in lines:
            segments = line.split(" ")

            input_values_raw = [x for x in segments[0]]

            _inputs = [int(x) for x in input_values_raw]
            _output = int(segments[1])
            data.append([_inputs, _output])

    random.shuffle(data)

    inputs = np.array([x[0] for x in data])
    outputs = np.array([x[1] for x in data])

    ratio = 0.75
    train_size = int(len(data) * ratio)
    train_inputs = inputs[:train_size]
    train_outputs = outputs[:train_size]
    test_inputs = inputs[train_size:]
    test_outputs = outputs[train_size:]

    ic("SGDClassifier")

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=60)
    clf.fit(train_inputs, train_outputs)

    sgd_correct = 0
    sgd_wrong = 0

    for index in range(train_size, len(inputs)):
        out = clf.predict([inputs[index]])
        if out[0] == outputs[index]:
            sgd_correct += 1
        else:
            sgd_wrong += 1

    ic(sgd_correct)
    ic(sgd_wrong)

    ic("Gaussian Process Regression ")

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel, random_state=0
    ).fit(train_inputs, train_outputs)
    gpr.score(train_inputs, train_outputs)

    gpr_correct = 0
    gpr_wrong = 0

    for index in range(train_size, len(inputs)):
        out = gpr.predict([inputs[index]], return_std=True)

        if out[0] > 0.5:
            gpr_correct += 1
        else:
            gpr_wrong += 1

    ic(gpr_correct)
    ic(gpr_wrong)

    ic("Multi layer neural network implementation")
    clf = MLPClassifier(
        solver='lbfgs', alpha=1e-5,
        hidden_layer_sizes=(5, 2), random_state=1
    )
    clf.fit(train_inputs, train_outputs)

    nn_correct = 0
    nn_wrong = 0

    for index in range(train_size, len(inputs)):
        out = clf.predict([inputs[index]])
        if out[0] == outputs[index]:
            nn_correct += 1
        else:
            nn_wrong += 1

    ic(nn_correct)
    ic(nn_wrong)

    ic("Decision Tree Classification")

    tree_classifier = tree.DecisionTreeClassifier()
    tree_classifier = tree_classifier.fit(inputs[:1400], outputs[:1400])
    tree_correct = 0
    tree_wrong = 0

    for index in range(train_size, len(inputs)):
        out = tree_classifier.predict([inputs[index]])
        if out == outputs[index]:
            tree_correct += 1
        else:
            tree_wrong += 1

    ic(tree_correct)
    ic(tree_wrong)


if __name__ == '__main__':
    main("datasets/data1.txt",)
