from neuro.neural_pattern import NeuralPattern

GLOBAL_COUNTER = 0


class PatternsConnection:

    def __init__(self,
                 source: NeuralPattern,
                 target: NeuralPattern,
                 agent,
                 area,
                 tick: int = 0,
                 weight: float = 1.0,
                 dope_value: int = 0,
                 ):
        self.source = source
        self.target = target
        self.weight = weight
        self.agent = agent
        self.tick = tick
        self.area = area
        self.dope_value = dope_value
        global GLOBAL_COUNTER
        self._id = GLOBAL_COUNTER
        GLOBAL_COUNTER += 1

    @classmethod
    def add(cls, source, target, agent, **kwargs) -> 'PatternsConnection':
        connection = cls(source, target, agent, **kwargs)
        agent.container.add_patterns_connection(connection)
        return connection

    def update_weight(self, val: float):
        change = 'increased' if val > 0 else 'decreased'
        log_info = f'connection ({self._id}) weight {change} from {self.weight} to {self.weight + val}'
        self.agent.logger.write_content(log_info)
        self.weight += val
        self.weight = min(1.0, self.weight)
        self.weight = max(0.1, self.weight)

    def _repr(self):
        return f'({self._id}) {self.area.name} val={self.target.data} source={self.source}'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()