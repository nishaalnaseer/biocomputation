from typing import List
import numpy
from icecream import ic
from random import shuffle, randint

from sklearn import tree
from sklearn.impute import KNNImputer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier


class Case:
    def __init__(self, inputs: list[float], output: int):
        self.inputs: list[float] = inputs
        self.output: int = output

    def __repr__(self) -> str:
        return f"{self.inputs}, {self.output}"


class TestedCase:
    def __init__(self, case: Case, distance: float):
        self.case: Case = case
        self.distance: float = distance


class TestCase:
    def __init__(self, case: Case):
        self.case: Case = case
        self.tested: List[TestedCase] = []

    def sort(self):
        sorted_tested = sorted(self.tested, key=lambda x: x.distance)
        self.tested = sorted_tested

    def calculate_output(self, k: int) -> int:
        neighbours = self.tested[:k]

        ones = 0
        zeros = 0

        for neighbour in neighbours:
            if neighbour.case.output == 1:
                ones += 1
            else:
                zeros += 1

        if ones > zeros:
            return 1
        else:
            return 0


def calculate_distance(train_case: Case, test_case: Case):
    total = 0
    for x in range(4):
        diff = train_case.inputs[x] - test_case.inputs[x]
        total += numpy.square(diff)

    return numpy.sqrt(total)


def load_data() -> list[Case]:
    cases: list[Case] = []
    with open("datasets/dataset3b.csv", 'r') as file:
        lines = file.readlines()
        lines.pop(0)

        for line in lines:
            segments = line.split(",")

            input_values_raw = [x for x in segments]
            inputs = [float(x) for x in input_values_raw]
            output = int(segments[-1])
            inputs.pop()
            case = Case(inputs, output)
            cases.append(case)

    shuffle(cases)

    return cases


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)


def predict(X, weights, bias):
    return sigmoid(numpy.dot(X, weights) + bias)


def main():
    # cases: list[Case] = load_data()
    # train_size = int(len(cases) * 0.75)
    # test_size = len(cases) - train_size
    # train_cases = cases[:train_size]
    # test_cases = cases[train_size:]
    #
    # k_values = [1, 3, 5, 7, 9]
    # ic(f"KNN implementation with k as {k_values}")
    # ic(f"Number of train cases {train_size}")
    # ic(f"Number of test cases {test_size}")
    # tested_cases: List[TestCase] = []
    #
    # for k in k_values:
    #     ic(f"With K as {k}")
    #
    #     correct_cases = 0
    #     wrong_cases = 0
    #     for test_case in test_cases:
    #         tested_case = TestCase(test_case)
    #         for train_case in train_cases:
    #             distance = calculate_distance(train_case, test_case)
    #             tested_case.tested.append(TestedCase(train_case, distance))
    #
    #         tested_cases.append(tested_case)
    #
    #         output = tested_case.calculate_output(k)
    #
    #         if output == test_case.output:
    #             correct_cases += 1
    #         else:
    #             wrong_cases += 1
    #
    #     ic(f"K as {k}: {correct_cases}")
    #     ic(f"K as {k}: {wrong_cases}")

    ic("stochastic gradient descent learning")

    data = []
    with open("datasets/dataset3b.csv", 'r') as file:
        lines = file.readlines()
        lines.pop(0)

        for line in lines:
            values = line.split(",")
            formatted = [float(values[index]) for index in range(0, len(values)-1)]
            formatted.append(int(values[-1]))
            data.append(formatted)

    train_size = int(len(data)*0.75)
    test_size = len(data)-train_size

    data = numpy.array(data)
    inputs = data[:, :-1]
    outputs = data[:, -1]

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=200)
    clf = clf.fit(inputs[:1400], outputs[:1400])

    sgd_correct = 0
    sgd_wrong = 0

    for index in range(1400, len(inputs)):
        out = clf.predict([inputs[index]])
        if out[0] == outputs[index]:
            sgd_correct += 1
        else:
            sgd_wrong += 1

    ic(sgd_correct)
    ic(sgd_wrong)

    ic("Decision Tree Classification")

    tree_classifier = tree.DecisionTreeClassifier()
    tree_classifier = tree_classifier.fit(inputs[:1400], outputs[:1400])
    tree_correct = 0
    tree_wrong = 0

    for index in range(1400, len(inputs)):
        out = tree_classifier.predict([inputs[index]])
        if out == outputs[index]:
            tree_correct += 1
        else:
            tree_wrong += 1

    ic(tree_correct)
    ic(tree_wrong)

    ic("Multi layer neural network implementation")
    clf = MLPClassifier(
        solver='adam', alpha=1e-5,
        hidden_layer_sizes=(10, 10), random_state=1,
        # max_iter=600
    )
    clf.fit(inputs[:1400], outputs[:1400])

    nn_correct = 0
    nn_wrong = 0

    for index in range(1400, len(inputs)):
        out = clf.predict([inputs[index]])
        if out[0] == outputs[index]:
            nn_correct += 1
        else:
            nn_wrong += 1

    ic(nn_correct)
    ic(nn_wrong)



if __name__ == '__main__':
    for x in range(3):
        main()
