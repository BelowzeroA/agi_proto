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
