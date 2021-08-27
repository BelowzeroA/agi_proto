from neuro.neural_pattern import NeuralPattern


class PatternsConnection:

    def __init__(self, source: NeuralPattern, target: NeuralPattern):
        self.source = source
        self.target = target
        self.weight = 0
        self.tick = 0