import random

from neuro.areas.receptive_area import ReceptiveArea
from neuro.neural_pattern import NeuralPattern

ANGLE_MARGINS = [22.5, 45.0, 67.5, 90.0, 110.5, 135.0, 157.5, 180.0]
NUM_SECTORS = 4
NEURAL_SPACE_SIZE = len(ANGLE_MARGINS) * 3 * NUM_SECTORS


class PrimitivesReceptiveArea(ReceptiveArea):
    """
    A receptive area for body shapes
    """
    def __init__(self, name: str, agent, zone):
        super().__init__(name, agent, zone)
        self.output_space_size = NEURAL_SPACE_SIZE

    def _categorize_angle(self, angle: float):
        for i in range(len(ANGLE_MARGINS)):
            if angle <= ANGLE_MARGINS[i]:
                return i
        raise AttributeError(f'Invalid angle: {angle}')

    def encode(self, data):
        neural_indices = []
        quadrant_index_space = int(NEURAL_SPACE_SIZE / NUM_SECTORS)
        for i, quadrant in enumerate(data['quadrants']):
            angle_masses = [0] * len(ANGLE_MARGINS)
            for segment in quadrant:
                angle_category = self._categorize_angle(segment['angle'])
                if angle_masses[angle_category] < 2:
                    angle_masses[angle_category] += segment['mass']
                    angle_masses[angle_category] = min(2, angle_masses[angle_category])

            for angle_idx, angle_category in enumerate(angle_masses):
                index = i * quadrant_index_space + (angle_idx * 3) + angle_category
                neural_indices.append(index)

        # neural_indices = list(set(neural_indices))
        # neural_indices.sort()
        return neural_indices

    def activate_on_roi(self, data):
        neural_indices = self.encode(data)
        neurons = self.container.get_area_neurons(self)
        for neuron in neurons:
            neuron.initially_active = False

        for idx in neural_indices:
            neurons[idx].initially_active = True

    def activate_on_body(self, data, name):
        neural_indices = self.encode(data)
        pattern = NeuralPattern.find_or_create(space_size=NEURAL_SPACE_SIZE, value=neural_indices)
        pattern.data = {self.name: name}
        self.output = pattern
