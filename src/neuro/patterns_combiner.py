import itertools

from neuro.neural_area import NeuralArea
from neuro.sdr_processor import SDRProcessor


class PatternsCombiner(NeuralArea):
    """
    A special kind of neural area that makes combinations of input patterns
    """
    def __init__(self, agent):
        self.agent = agent
        self.output_areas = []
        self.inputs = []
        self.input_sizes = []
        self.last_reset_tick = 0

    def combine_transfer(self):
        alive_patterns = [pattern for pattern in self.inputs if pattern]
        if len(alive_patterns) == 0:
            for area in self.output_areas:
                area.inputs = [None]
            return

        combinations = [alive_patterns]
        for i in range(len(alive_patterns) - 1):
            combinations.extend(itertools.combinations(alive_patterns, i + 1))

        patterns = []
        for combination in combinations:
            if 3 > len(combination) > 1:
                input_sizes = [p.space_size for p in combination]
                pattern = SDRProcessor.make_combined_pattern(combination, input_sizes)
            elif len(combination) == 1:
                pattern = combination[0]
            else:
                continue
            patterns.append(pattern)

        for area in self.output_areas:
            area.receive_inputs(patterns)

        self.reset_inputs()

    def reset_inputs(self):
        current_tick = self.agent.network.current_tick
        if current_tick - self.last_reset_tick > 10:
            self.inputs = [None for i in range(len(self.input_sizes))]
            self.last_reset_tick = current_tick


