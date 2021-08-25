from typing import List, Union

from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.sdr_processor import SDRProcessor


class EncoderArea(NeuralArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
            output_space_size: int,
            output_norm: int,
            min_inputs: int = 1,
            surprise_level: int = 1,
    ):
        super().__init__(name=name, agent=agent, zone=zone)
        self.output_space_size = output_space_size
        self.output_norm = output_norm
        self.min_inputs = min_inputs
        self.processor = SDRProcessor(self)
        self.patterns: List[NeuralPattern] = []
        self.highway_connections = set()
        self.connections = []
        self.surprise_level = surprise_level

    def update(self):
        self.output = None
        alive_inputs = len([pattern for pattern in self.inputs if pattern])
        if alive_inputs < self.min_inputs:
            self.reset_inputs()
            return

        combined_pattern = SDRProcessor.make_combined_pattern(self.inputs, self.input_sizes)
        if combined_pattern:
            if self.container.network.verbose:
                print(f'Combined receptive pattern: {combined_pattern}')
            self.process_input(combined_pattern)
        else:
            self.output = None
        self.reset_inputs()

        super().update()

    def reset_inputs(self):
        self.inputs = [None for i in range(len(self.input_sizes))]

    def process_input(self, pattern: NeuralPattern) -> None:
        output_pattern, pattern_is_new = self.processor.process_input(pattern)
        output_pattern.data = pattern.data
        output_pattern.log(self)
        if pattern_is_new:
            if self.container.network.verbose:
                print(f'[{self.name}]: New pattern has been created {output_pattern}')
            if self.surprise_level > 0:
                agent = self.container.network.agent
                agent.on_message({
                    'message': 'pattern_created',
                    'surprise_level': self.surprise_level,
                    'area': self,
                    'pattern': output_pattern
                })
        else:
            self.output = output_pattern
            if self.container.network.verbose:
                print(f'[{self.name}]: Existing pattern has been recognized {output_pattern}')
