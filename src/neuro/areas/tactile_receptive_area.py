from neuro.areas.receptive_area import ReceptiveArea
from neuro.neural_pattern import NeuralPattern
from neuro.sdr_processor import SDRProcessor


class TactileReceptiveArea(ReceptiveArea):
    """
    Encodes tactile observations
    """
    def __init__(self, name: str, agent, zone):
        super().__init__(name, agent, zone)
        self.output_space_size = 200
        self.single_space_size = 100
        self.output_norm = 20
        self.patterns_clenched = {}
        self.patterns_holding = {}
        self._generate_patterns()

    def _make_random_pattern(self, data: dict):
        pattern = NeuralPattern(space_size=self.single_space_size, value_size=self.output_norm // 2)
        pattern.generate_random()
        pattern.data = data
        return pattern

    def _generate_patterns(self):
        self.patterns_clenched[False] = self._make_random_pattern({'clenched': False})
        self.patterns_clenched[True] = self._make_random_pattern({'clenched': True})
        self.patterns_holding[False] = self._make_random_pattern({'holding': False})
        self.patterns_holding[True] = self._make_random_pattern({'holding': True})

    def activate(self, data):
        pattern_clenched = self.patterns_clenched[data['is_clenched']]
        pattern_holding = self.patterns_holding[data['is_holding']]
        self.output = SDRProcessor.make_combined_pattern(
            [pattern_clenched, pattern_holding], [self.single_space_size] * 2)
