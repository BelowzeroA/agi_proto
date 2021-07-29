from neuro.neural_area import NeuralArea


class Neuron:

    def __init__(self, id: str, area: NeuralArea, container):
        self.id = id
        self.threshold = 2
        self.area = area
        self.container = container

