import collections
import random
from typing import List, Union

from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.sdr_processor import SDRProcessor


class EncoderArea(NeuralArea):

    def __init__(
            self,
            name: str,
            output_space_size: int,
            output_activity_norm: int,
            container,
            min_inputs: int = 1
    ):
        super().__init__(name=name, container=container)
        self.output_space_size = output_space_size
        self.output_activity_norm = output_activity_norm
        self.min_inputs = min_inputs
        self.processor = SDRProcessor(self)
        self.patterns: List[NeuralPattern] = []
        self.highway_connections = set()
        self.connections = []

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

    def process_input(self, pattern: NeuralPattern) -> None:
        output_pattern = self.processor.process_input(pattern)
        if not self.output:
            if self.container.network.verbose:
                print(f'[{self.name}]: New pattern has been created {output_pattern}')
            agent = self.container.network.agent
            agent.on_message('pattern_created')
        else:
            if self.container.network.verbose:
                print(f'[{self.name}]: Existing pattern has been recognized {output_pattern}')
