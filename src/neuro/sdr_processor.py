import collections
import random
from typing import Union, List

from neuro.hyper_params import HyperParameters
from neuro.neural_pattern import NeuralPattern


class SDRProcessor:

    def __init__(self, area: 'EncoderArea'):
        self.area = area

    def process_input(self, pattern: NeuralPattern) -> NeuralPattern:
        output, connections = self._get_raw_output(pattern)
        output_pattern = None
        for attempted_pattern in self.area.patterns:
            if self.patterns_similar(attempted_pattern, output):
                output_pattern = attempted_pattern
                break

        pattern_is_new = False
        if not output_pattern:
            pattern_is_new = True
            output_pattern = NeuralPattern(self.area.output_space_size, value=output)
            self.area.patterns.append(output_pattern)
            self.area.highway_connections.update(connections)
        return output_pattern, pattern_is_new

    def patterns_similar(self, pattern1: NeuralPattern, pattern2: Union[NeuralPattern, List[int]]):
        value = pattern2
        if isinstance(pattern2, NeuralPattern):
            value = pattern2.value
        intersection = set(pattern1.value) & set(value)
        return len(intersection) >= self.area.output_norm * HyperParameters.pattern_recognition_threshold

    def _get_raw_output(self, pattern: NeuralPattern) -> NeuralPattern:
        connections = self.area.connections

        if len(connections) == 0:
            output_space_indexes = list(range(self.area.output_space_size))

            ratio = 0.15 * (self.area.output_space_size * self.area.output_norm) / pattern.value_size
            connection_density = int(ratio)
            for idx in range(pattern.space_size):
                connections.append(random.sample(output_space_indexes, connection_density))

        output, highway_connections = self._select_pattern(pattern, connections)
        if len(output) > self.area.output_norm:
            output = random.sample(output, self.area.output_norm)

        output.sort()
        return output, highway_connections

    def _select_pattern(self, pattern: NeuralPattern, connections):
        while True:
            out_pattern, activated_connections = self._sample_activations(pattern, connections)
            if out_pattern:
                activated_connections = [c for c in activated_connections if c[1] in out_pattern]
                return out_pattern, activated_connections

    def _sample_activations(self, pattern: NeuralPattern, connections):
        output_norm = self.area.output_norm
        target_cells_activated = []
        activated_connections = []
        for idx in pattern.value:
            outgoing_connections = connections[idx]
            outgoing_activations = [c for c in outgoing_connections if random.randint(0, 9) < 1]
            highway_activations = [c[1] for c in self.area.highway_connections if c[0] == idx]
            outgoing_activations.extend(highway_activations * 2)
            activated_connections.extend([(idx, target) for target in outgoing_activations])
            target_cells_activated.extend(outgoing_activations)

        counters = list(collections.Counter(target_cells_activated).items())
        counters.sort(key=lambda x: x[1], reverse=True)
        highest_potential = counters[0][1]
        if highest_potential > 2:
            result = [c[0] for c in counters if c[1] > 2]
            if len(result) >= output_norm:
                return result[:output_norm], activated_connections
            else:
                with_high_potential = [c[0] for c in counters if c[1] == 2]
                if len(result) + len(with_high_potential) >= output_norm:
                    result = result + random.sample(with_high_potential, output_norm - len(result))
                    return result, activated_connections
                else:
                    return None, None
        else:
            result = [c[0] for c in counters if c[1] > 1]
            if len(result) >= self.area.output_norm:
                return result, activated_connections
        return None, None

    @staticmethod
    def make_combined_pattern(inputs: List[NeuralPattern], input_sizes) -> NeuralPattern:
        combined_input_indices = []
        combined_input_data = {}
        combined_pattern = None
        histories = []
        shift = 0
        for i in range(len(inputs)):
            cur_input = inputs[i]
            if cur_input:
                combined_input_indices.extend([idx + shift for idx in cur_input.value])
                histories.append(cur_input.history)
                if cur_input.data:
                    for key in cur_input.data:
                        combined_input_data[key] = cur_input.data[key]
            shift += input_sizes[i]

        if len(combined_input_indices):
            combined_pattern = NeuralPattern(
                space_size=sum(input_sizes),
                value=combined_input_indices,
                data=combined_input_data
            )
            combined_pattern.merge_histories(histories)
        return combined_pattern

