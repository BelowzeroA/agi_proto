from typing import List, Union

from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern


class ActionArea(NeuralArea):

    def __init__(
            self,
            name: str,
            action_id: str,
            output_space_size: int,
            output_activity_norm: int,
            container,
    ):
        super().__init__(name=name, container=container)
        self.action_id = action_id
        self.output_space_size = output_space_size
        self.output_activity_norm = output_activity_norm
        self.patterns: List[NeuralPattern] = []

    def update(self):
        self.output = None
        alive_inputs = len([pattern for pattern in self.inputs if pattern])
        if alive_inputs < self.min_inputs:
            self.reset_inputs()
            return

        combined_input = []
        shift = 0
        for i in range(len(self.inputs)):
            if self.inputs[i]:
                combined_input.extend([idx + shift for idx in self.inputs[i].value])
            shift += self.input_sizes[i]

        if len(combined_input):
            combined_pattern = NeuralPattern(space_size=sum(self.input_sizes), value=combined_input)
            if self.container.network.verbose:
                print(f'Combined receptive pattern: {combined_pattern}')
            self.process_input(combined_pattern)
        else:
            self.output = None
        self.reset_inputs()

    def reset_inputs(self):
        self.inputs = [None for i in range(len(self.input_sizes))]
