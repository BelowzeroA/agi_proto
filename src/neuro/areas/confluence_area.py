from typing import List, Union

from neuro.areas.encoder_area import EncoderArea
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection
from neuro.sdr_processor import SDRProcessor


class ConfluenceArea(EncoderArea):
    """
    Makes the merging of two sparse patterns
    """
    def __init__(
            self,
            name: str,
            agent,
            zone,
    ):
        super().__init__(name=name, agent=agent, zone=zone)
        self.min_inputs = 2

    def update(self):
        super().update()
