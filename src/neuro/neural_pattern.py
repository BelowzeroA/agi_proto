from typing import List

from neuro.neuron import Neuron


class NeuralPattern:

    def __init__(self, neurons: List[Neuron], current_tick: int):
        self.neurons = set(neurons)
        self.start_activity_tick = current_tick
        self.active = True

    def intersection(self, neurons):
        return self.neurons.intersection(neurons)

    def is_in(self, neuron: Neuron):
        return neuron in self.neurons