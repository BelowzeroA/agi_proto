import collections
import random
from typing import List, Union

from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.sdr_processor import SDRProcessor


class EncoderArea(NeuralArea):

    def __init__(self, name: str, output_space_size: int, output_activity_norm: int, container):
        super().__init__(name=name, container=container)
        self.output_space_size = output_space_size
        self.output_activity_norm = output_activity_norm
        self.processor = SDRProcessor(self)
        self.patterns: List[NeuralPattern] = []
        self.highway_connections = set()

    def update(self):
        self.output = None
        combined_input = []
        shift = 0
        for i in range(len(self.inputs)):
            if self.inputs[i]:
                combined_input.extend([idx + shift for idx in self.inputs[i].value])
            shift += self.input_sizes[i]

        if len(combined_input):
            combined_pattern = NeuralPattern(space_size=sum(self.input_sizes), value=combined_input)
            print(f'Combined receptive pattern: {combined_pattern}')
            self.process_input(combined_pattern)
        else:
            self.output = None
        # print(self.output)
        self.inputs = [None for i in range(len(self.input_sizes))]

    def process_input(self, pattern: NeuralPattern) -> NeuralPattern:
        output_pattern = self.processor.process_input(pattern)
        if not self.output:
            print(f'[{self.name}]: New pattern has been created {output_pattern}')
        else:
            print(f'[{self.name}]: Existing pattern has been recognized {output_pattern}')

[21, 77, 82, 119, 148, 150, 225, 259, 343, 407, 460, 514, 533, 554, 566, 567, 773, 811, 852, 885]
[21, 77, 98, 119, 150, 171, 225, 252, 259, 331, 403, 408, 514, 533, 574, 773, 819, 820, 885, 890]