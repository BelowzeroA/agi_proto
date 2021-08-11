import random

from neuro.hyper_params import HyperParameters
from neuro.neuron import Neuron

LOWEST_WEIGHT = 0.1


class Connection_legacy:

    def __init__(self, source: Neuron, target: Neuron):
        self.source = source
        self.target = target
        self.pulsing = False
        self.has_pulsed = False
        self.weight = 0.2

    def update(self):
        if self.pulsing:
            margin = int(100 * self.weight)
            rand_val = random.randint(1, 100)
            if rand_val <= margin:
                self.target.potential += 1
                self.has_pulsed = True
                self.target.add_to_history(self)
        self.pulsing = False

    def update_weight(self):
        if self.has_pulsed:
            if self.source.firing and self.target.firing:
                self.weight += HyperParameters.learning_rate
            else:
                self.weight -= HyperParameters.learning_rate
        if self.weight > 1:
            self.weight = 1
        elif self.weight <= LOWEST_WEIGHT:
            self.weight = LOWEST_WEIGHT

    def _repr(self):
        return f'{self.source} - {self.target}]'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()