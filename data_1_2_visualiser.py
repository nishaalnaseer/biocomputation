import matplotlib.pyplot as plt
import numpy as np

def calculate_distance(individual1, individual2):
    # Placeholder for your distance calculation function
    return np.linalg.norm(np.array(individual1) - np.array(individual2))

data = []
filename = 'datasets/data1.txt'  # Replace with your actual filename

with open(filename, 'r') as file:
    lines = file.readlines()
    lines.pop(0)

    for line in lines:
        segments = line.split(" ")

        input_values_raw = [x for x in segments[0]]

        _inputs = [int(x) for x in input_values_raw]
        _output = int(segments[1])
        data.append([_inputs, _output])

inputs = np.array([x[0] for x in data])
outputs = np.array([x[1] for x in data])

index = 0
for _index, output in enumerate(outputs):
    if output == 1:
        index = _index
        break
individual1 = inputs[index]
individual0 = inputs[1]

individual0_distances = [calculate_distance(individual0, x) for x, y in data]
individual1_distances = [calculate_distance(individual1, x) for x, y in data]

individual0_finished = []
individual1_finished = []

for index in range(len(individual0_distances)):
    individual0_finished.append([individual0_distances[index], outputs[index]])

for index in range(len(individual1_distances)):
    individual1_finished.append([individual1_distances[index], outputs[index]])

colours = ['red', 'blue']

fig, ax = plt.subplots()
count = 1
for _index in range(len(individual0_finished)):
    distance = individual0_finished[_index][0]
    out = individual0_finished[_index][1]

    ax.text(distance, count, f'{distance:.2f}', ha='center', va='center')
    ax.add_patch(
        plt.Rectangle((distance - 0.5, count - 0.5), 1, 1, fill=True, color=colours[out])
    )
    count += 1

ax.set_xlim(-0.5, max(individual0_distances) + 0.5)
ax.set_ylim(0.5, count)
ax.set_aspect('auto')

# Hide axes
ax.axis('off')

plt.show()
