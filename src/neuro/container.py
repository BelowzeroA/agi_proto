from typing import List

from neuro.connection import Connection
from neuro.neural_area import NeuralArea
from neuro.neuron import Neuron


class NeuroContainer:

    def __init__(self):
        self.neurons = []
        self.areas = []
        self.connections = []

    def add_area(self, area):
        self.areas.append(area)

    def get_area_neurons(self, area: NeuralArea) -> List[Neuron]:
        return [n for n in self.neurons if n.area == area]

    def next_neuron_id(self):
        if len(self.neurons) == 0:
            return '1'
        return str(max([int(neuron.id) for neuron in self.neurons]) + 1)

    def add_neuron(self, area) -> Neuron:
        neuron = Neuron(id=self.next_neuron_id(), area=area, container=self)
        self.neurons.append(neuron)
        return neuron

    def add_connection(self, source: Neuron, target: Neuron) -> Connection:
        conn = Connection(source=source, target=target)
        self.connections.append(conn)
        return conn