import random


class NeuralArea:

    def __init__(self, name: str, container):
        self.name = name
        self.container = container

    def connect(self, area: 'NeuralArea', density: int):
        source_neurons = self.container.get_area_neurons(self)
        target_neurons = self.container.get_area_neurons(area)
        for source_neuron in source_neurons:
            target_neurons_selection = random.sample(target_neurons, n=density)


