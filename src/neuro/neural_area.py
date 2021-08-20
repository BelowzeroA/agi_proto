from neuro.neural_pattern import NeuralPattern


class NeuralArea:
    """
    Common data and methods for all neural areas
    """
    def __init__(self, name: str, agent, zone):
        self.agent = agent
        self.container = agent.container
        self.name = name
        self.inputs = []
        self.input_sizes = []
        self.output: NeuralPattern = None
        self.output_space_size = 0
        self.zone = zone

    @classmethod
    def add(cls, name, agent, zone, **kwargs) -> 'NeuralArea':
        area = cls(name, agent, zone, **kwargs)
        agent.container.add_area(area)
        zone.areas.append(area)
        return area

    def update(self):
        self.zone.on_area_updated(self)

    def report(self):
        pass

    def _repr(self):
        return f'[{self.name}]'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()


