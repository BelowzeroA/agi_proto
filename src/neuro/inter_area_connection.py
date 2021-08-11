from neuro.neural_area import NeuralArea


class InterAreaConnection:

    def __init__(self, source: NeuralArea, target: NeuralArea):
        self.source = source
        self.target = target
        self.is_open = True
        self.target_slot_index = None

    def update(self):
        if self.is_open:
            if self.source.output:
                self.target.inputs[self.target_slot_index] = self.source.output

    def on_adding(self):
        self.target_slot_index = len(self.target.inputs)
        self.target.inputs.append(None)
        self.target.input_sizes.append(self.source.output_space_size)