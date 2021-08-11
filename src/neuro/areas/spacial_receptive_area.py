import random

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern

NEURAL_SPACE_SIZE = 64


class SpacialReceptiveArea(NeuralArea):

    def __init__(self, name: str, container, output_space_size: int = None, output_activity_norm: int = None):
        super().__init__(name, container)
        self.output_space_size = output_space_size or HyperParameters.space_encoder_space_size
        self.output_activity_norm = output_activity_norm or HyperParameters.space_encoder_activity_norm
        self.pattern_cache = {}

    def encode(self, data: float):
        if data == -1:
            return None

        assert 0 <= data <= 1, 'spacial data must be normalized'

        data_hash = int(data * 100)
        if data_hash in self.pattern_cache:
            return self.pattern_cache[data_hash]

        center_index = int(self.output_space_size * data)
        probabilities = []
        max_distance_from_center = self.output_space_size - center_index
        for i in range(self.output_space_size):
            distance_from_center = i - center_index
            if distance_from_center < 0:
                distance_from_center = -distance_from_center
            if distance_from_center > 1:
                probabilities.extend([i] * ((max_distance_from_center - distance_from_center) ** 2))

        indices = [center_index]
        if center_index > 0:
            indices.append(center_index - 1)
        if center_index < self.output_space_size - 1:
            indices.append(center_index + 1)

        while len(indices) < self.output_activity_norm:
            index = random.choice(probabilities)
            if index not in indices:
                indices.append(index)
        indices.sort()
        self.pattern_cache[data_hash] = indices
        return indices

    def activate_on_body(self, data):
        neural_indices = self.encode(data)
        if neural_indices:
            pattern = NeuralPattern(space_size=self.output_space_size, value=neural_indices)
        else:
            pattern = None
        self.output = pattern
