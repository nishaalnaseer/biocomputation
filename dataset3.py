from random import shuffle

from icecream import ic


def main():
    data = {}
    test = []
    with open("datasets/dataset3b.csv", 'r') as file:
        lines = file.readlines()
        lines.pop(0)

        _content = []

        for line in lines:
            segments = line.split(",")
            _content.append(
                [
                    "".join(segments[:-1]),
                    segments[-1].strip()
                 ]
            )
        shuffle(_content)

        train_size = int(len(_content) * 0.75)
        train = _content[:train_size]
        test = _content[train_size:]
        _data = {row[0]: row[1] for row in train}
        data = _data

    data = {i: data[i] for i in sorted(data, key=lambda x: int(x))}
    ic(data)

    for index in range(len(test)):
        content = test[index]

        _input = content[0]
        _output = content[1]
        predicted = data[_input]
        if predicted != _output:
            raise Exception("Unexpected")


if __name__ == '__main__':
    main()
