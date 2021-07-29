from neuro.neuron import Neuron


class Connection:

    def __init__(self, source: Neuron, target: Neuron):
        self.source = source
        self.target = target