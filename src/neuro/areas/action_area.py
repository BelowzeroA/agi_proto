from typing import List, Union

from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern


class ActionArea(NeuralArea):

    def __init__(
            self,
            name: str,
            action_id: str,
            output_space_size: int,
            output_norm: int,
            container,
    ):
        super().__init__(name=name, container=container)
        self.action_id = action_id
        self.output_space_size = output_space_size
        self.output_norm = output_norm
        self.patterns: List[NeuralPattern] = []

    def update(self):
        pass
