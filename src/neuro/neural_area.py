from neuro.neural_pattern import NeuralPattern


class NeuralArea:

    def __init__(self, name: str, container):
        self.container = container
        self.name = name
        self.inputs = []
        self.input_sizes = []
        self.output: NeuralPattern = None
        self.output_space_size = 0

    def update(self):
        pass

    def report(self):
        pass



