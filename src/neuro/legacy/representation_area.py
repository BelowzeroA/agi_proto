import random

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.neuron import Neuron


class RepresentationArea(NeuralArea):

    def __init__(self, name: str, container):
        super().__init__(name, container)
        self.patterns = []
        self.neurons = []
        self.active_pattern: NeuralPattern = None
        self.allocate()

    def allocate(self):
        for i in range(HyperParameters.representation_neurons_num):
            neuron = self.container.add_neuron(area=self)
            self.neurons.append(neuron)

    def firing_allowed(self, neuron: Neuron):
        if self.active_pattern:
            return self.active_pattern.is_in(neuron)
        return True

    def firing_mandatory(self, neuron: Neuron):
        if self.active_pattern:
            return self.active_pattern.is_in(neuron)
        return False

    def update(self):
        current_tick = self.container.network.current_tick
        if self.active_pattern:
            if current_tick - self.active_pattern.start_activity_tick > HyperParameters.pattern_longevity:
                self.active_pattern = None
            else:
                return

        active_neurons = [n for n in self.neurons if n.firing]
        min_detectable_neurons_num = int(HyperParameters.active_neurons_ratio * len(self.neurons))
        min_pattern_recognition_neurons_num = int(HyperParameters.recognition_ratio * min_detectable_neurons_num)

        if len(active_neurons) < min_detectable_neurons_num:
            return

        if len(active_neurons) > min_detectable_neurons_num:
            active_neurons = random.sample(active_neurons, min_detectable_neurons_num)

        recognized_pattern = None
        for pattern in self.patterns:
            intersection = pattern.intersection(active_neurons)
            if len(intersection) >= min_pattern_recognition_neurons_num:
                recognized_pattern = pattern
                break

        if recognized_pattern:
            activity_period = current_tick - recognized_pattern.start_activity_tick
            if HyperParameters.pattern_longevity + 3 > activity_period > HyperParameters.pattern_longevity:
                recognized_pattern.active = False
                return
            self.active_pattern = recognized_pattern
            self.active_pattern.start_activity_tick = current_tick
        else:
            self.active_pattern = NeuralPattern(active_neurons, current_tick)
            self.patterns.append(self.active_pattern)




