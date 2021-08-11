
import random

from neuro.hyper_params import HyperParameters
from neuro.container import Container
from neuro.encoder_area import EncoderArea
from neuro.network import Network
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern


def make_firing_pattern(container, area, coefficient=0.2):
    initially_active_neurons_num = int(HyperParameters.receptive_neurons_num * coefficient)
    neurons = container.get_area_neurons(area)
    selected = random.sample(neurons, initially_active_neurons_num)
    return selected


def change_firing_pattern(pattern, container, area, coefficient):
    num_neurons_to_be_replaced = int(len(pattern) * coefficient)
    neurons = container.get_area_neurons(area)
    new_pattern = random.sample(pattern, len(pattern) - num_neurons_to_be_replaced)
    while len(new_pattern) < len(pattern):
        neuron = random.choice(neurons)
        if neuron not in pattern and neuron not in new_pattern:
            new_pattern.append(neuron)
    return new_pattern


def main():
    random.seed(0)
    container = Container()
    receptive_area = NeuralArea(name='receptive')
    presentation_area = EncoderArea(name='representations', output_space_size=1000, output_activity_norm=20)

    container.add_area(receptive_area)
    container.add_area(presentation_area)

    container.add_connection(source=receptive_area, target=presentation_area)

    network = Network(container=container)

    shift = 0.1

    pattern = NeuralPattern(space_size=64, value_size=7)
    pattern.generate_random()
    pattern.value = [1, 4, 8, 28, 35, 48, 58]

    receptive_area.output = pattern

    network.run(max_iter=2)

    pattern.value = [2, 5, 9, 29, 36, 49, 60]
    network.run(max_iter=2)


if __name__ == '__main__':
    main()
