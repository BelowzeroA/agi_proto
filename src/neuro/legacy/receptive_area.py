import random

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neuron import Neuron


class ReceptiveArea(NeuralArea):

    def __init__(self, name: str, container):
        super().__init__(name, container)
        self.allocate()

    def allocate(self):
        for i in range(HyperParameters.receptive_neurons_num):
            self.container.add_neuron(area=self)

    def activate_random_firing_pattern(self, coefficient=0.2):
        initially_active_neurons_num = int(HyperParameters.receptive_neurons_num * coefficient)
        neurons = self.container.get_area_neurons(self)
        for neuron in neurons:
            neuron.initially_active = False
        selected = random.sample(neurons, initially_active_neurons_num)
        for neuron in selected:
            neuron.initially_active = True

    def activate_firing_pattern(self, pattern):
        neurons = self.container.get_area_neurons(self)
        for neuron in neurons:
            neuron.initially_active = False
        for neuron in pattern:
            neuron.initially_active = True

