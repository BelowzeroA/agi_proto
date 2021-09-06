from neuro.neural_area import NeuralArea


class WorkingMemoryCell(NeuralArea):

    def __init__(self, name: str, agent, zone):
        super().__init__(name, agent, zone)
        self.intensity = 0

    def update(self):
        if len(self.inputs):
            self.output = self.inputs[0]
