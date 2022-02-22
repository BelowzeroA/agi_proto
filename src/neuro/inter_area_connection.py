from neuro.neural_area import NeuralArea


class InterAreaConnection:
    """
    Genetically determined connection between neural areas
    """
    def __init__(self, source: NeuralArea, target: NeuralArea, source_output_property: str = None):
        self.source = source
        self.target = target
        self.is_open = True
        self.target_slot_index = None
        self.source_output_property = source_output_property

    def update(self):
        if not self.is_open:
            return

        if self.source_output_property:
            output = getattr(self.source, self.source_output_property)
        else:
            output = self.source.output

        if output or self.target.receive_empty_input:
            self.target.inputs[self.target_slot_index] = output

    def on_adding(self):
        self.target_slot_index = len(self.target.inputs)
        self.target.inputs.append(None)
        self.target.input_sizes.append(self.source.output_space_size)