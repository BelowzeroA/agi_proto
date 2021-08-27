from typing import List, Union

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection
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
            recognition_threshold=None,
    ):
        super().__init__(name=name, agent=agent, zone=zone)
        self.output_space_size = output_space_size
        self.output_norm = output_norm
        self.min_inputs = min_inputs
        self.processor = SDRProcessor(self)
        self.patterns: List[NeuralPattern] = []
        self.highway_connections = set()
        self.connections = []
        self.pattern_connections = []
        self.surprise_level = surprise_level
        self.recognition_threshold = recognition_threshold or HyperParameters.pattern_recognition_threshold

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

    def recognize_output_pattern(self, input_pattern: NeuralPattern):
        for connection in self.pattern_connections:
            if connection.source.similarity(input_pattern) >= self.recognition_threshold:
                return connection.target
        return None

    def process_input0(self, pattern: NeuralPattern) -> None:
        recognized_output = self.recognize_output_pattern(pattern)
        if recognized_output:
            self.output = recognized_output
            return

        output_pattern, pattern_is_new = self.processor.process_input(pattern)
        output_pattern.data = pattern.data
        output_pattern.log(self)
        if pattern_is_new:
            output_pattern.source_patterns = list(self.inputs)
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

    def process_input(self, pattern: NeuralPattern) -> None:
        output_pattern, is_new = self.recognize_process_input(pattern)
        if is_new:
            output_pattern.source_patterns = list(self.inputs)
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

    def recognize_process_input(self, pattern: NeuralPattern) -> NeuralPattern:
        recognized_output = self.recognize_output_pattern(pattern)
        if recognized_output:
            return recognized_output, False

        output_pattern = self.processor.process_input(pattern)
        output_pattern.data = pattern.data
        output_pattern.log(self)

        connection = PatternsConnection(
            source=pattern,
            target=output_pattern
        )
        self.pattern_connections.append(connection)

        return output_pattern, True