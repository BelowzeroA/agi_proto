import random

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern


class ReceptiveArea(NeuralArea):
    """
    Abstract receptive area
    """
    def __init__(
            self,
            name: str,
            agent,
            zone,
    ):
        super().__init__(name, agent, zone)
        self.is_receptive = True

    def reset_output(self):
        self.output = None