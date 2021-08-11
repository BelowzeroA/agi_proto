import random

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neuron import Neuron

NEURAL_SPACE_SIZE = 64


class PrimitivesReceptiveArea(NeuralArea):

    def __init__(self, name: str, container):
        super().__init__(name, container)
        self.allocate()

    def allocate(self):
        for i in range(NEURAL_SPACE_SIZE):
            self.container.add_neuron(area=self)

    def _categorize_angle(self, angle: float):
        angle_margins = [22.5, 45.0, 67.5, 90.0, 110.5, 135.0, 157.5, 180.0]
        for i in range(len(angle_margins)):
            if angle <= angle_margins[i]:
                return i
        raise AttributeError(f'Invalid angle: {angle}')

    def encode(self, data):
        neural_indices = []
        quadrant_index_space = int(NEURAL_SPACE_SIZE / 4)
        for i, quadrant in enumerate(data['quadrants']):
            for segment in quadrant:
                angle_category = self._categorize_angle(segment['angle'])
                index_in_quadrant = angle_category * 2 # + segment['mass'] - 1
                index = i * quadrant_index_space + index_in_quadrant
                neural_indices.append(index)
                if segment['mass'] == 2:
                    neural_indices.append(index + 1)
        return neural_indices

    def activate_on_roi(self, data):
        neural_indices = self.encode(data)
        neurons = self.container.get_area_neurons(self)
        for neuron in neurons:
            neuron.initially_active = False

        for idx in neural_indices:
            neurons[idx].initially_active = True


[41], [42], [45], [46], [49], [50], [53],
[83], [94], [108], [121], [127], [131], [141], [143], [163], [164]

[1], [13], [14], [37], [43], [51]
[83], [94], [108], [121], [127], [131], [141], [143], [163], [164]

[1], [13], [14], [49], [53], [54], [57], [58]
[68], [85], [86], [97], [98], [114], [136], [140], [153], [157]

[21], [22], [23], [27], [49], [50], [61]
[86], [92], [93], [95], [100], [102], [120], [137], [145], [153]
