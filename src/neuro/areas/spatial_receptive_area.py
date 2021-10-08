import random

from neuro.areas.receptive_area import ReceptiveArea
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern


class SpatialReceptiveArea(ReceptiveArea):
    """
    Encodes continuous values such as distance, speed, etc.
    """
    def __init__(
            self,
            name: str,
            agent,
            zone,
            output_space_size: int = None,
            output_norm: int = None,
            grid_size: int = None
    ):
        super().__init__(name, agent, zone)
        self.output_space_size = output_space_size or HyperParameters.space_encoder_space_size
        self.output_norm = output_norm or HyperParameters.space_encoder_norm
        self.grid_size = grid_size or self.output_space_size
        assert self.grid_size <= self.output_space_size, 'grid_size must be less or equal to output_space_size'
        self.pattern_cache = {}

    def encode(self, data: float):
        if data == -1:
            return None

        if data > 0.99:
            data = 0.99
        assert 0 <= data < 1, 'spatial data must be normalized'

        data_hash = int(data * self.grid_size)
        if data_hash in self.pattern_cache:
            return self.pattern_cache[data_hash]

        center_index = int(self.output_space_size * data)
        probabilities = []
        space_center_index = self.output_space_size // 2
        if center_index <= space_center_index:
            max_distance_from_center = self.output_space_size - center_index
        else:
            max_distance_from_center = center_index

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

        while len(indices) < self.output_norm:
            index = random.choice(probabilities)
            if index not in indices:
                indices.append(index)
        indices.sort()
        self.pattern_cache[data_hash] = indices
        return indices

    def activate_on_body(self, data):
        neural_indices = self.encode(data)
        if neural_indices:
            pattern = NeuralPattern.find_or_create(
                space_size=self.output_space_size,
                value=neural_indices,
                data={self.name: data})
        else:
            pattern = None
        self.output = pattern
