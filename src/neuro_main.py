import random

from neuro.areas.representation_area import RepresentationArea
from neuro.areas.receptive_area import ReceptiveArea
from neuro.container import NeuroContainer
from neuro.hyper_params import HyperParameters
from neuro.network import Network
from neuro.neural_area import NeuralArea


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
    container = NeuroContainer()
    receptive_area = ReceptiveArea(name='receptive', container=container)
    presentation_area = RepresentationArea(name='representations', container=container)
    container.add_area(receptive_area)
    container.add_area(presentation_area)

    receptive_area.connect(presentation_area, density=15)

    network = Network(container=container)

    shift = 0.1

    firing_pattern = make_firing_pattern(container, receptive_area)
    receptive_area.activate_firing_pattern(firing_pattern)

    network.run(max_iter=50)

    network.reset()

    new_pattern = change_firing_pattern(firing_pattern, container, receptive_area, coefficient=shift)
    receptive_area.activate_firing_pattern(new_pattern)
    presentation_area.active_pattern = None
    network.run(max_iter=50)

    network.reset()
    new_pattern = change_firing_pattern(firing_pattern, container, receptive_area, coefficient=shift)
    receptive_area.activate_firing_pattern(new_pattern)
    presentation_area.active_pattern = None
    network.run(max_iter=50)

    network.reset()
    new_pattern = change_firing_pattern(firing_pattern, container, receptive_area, coefficient=shift)
    receptive_area.activate_firing_pattern(new_pattern)
    presentation_area.active_pattern = None
    network.run(max_iter=50)

    network.reset()
    new_pattern = change_firing_pattern(firing_pattern, container, receptive_area, coefficient=1.0)
    receptive_area.activate_firing_pattern(new_pattern)
    presentation_area.active_pattern = None
    network.run(max_iter=50)


if __name__ == '__main__':
    main()
