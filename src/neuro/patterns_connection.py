from neuro.hyper_params import HyperParameters
from neuro.neural_pattern import NeuralPattern

GLOBAL_COUNTER = 0


class PatternsConnection:
    """
    Connection between two patterns. According to the mainstream neuroscientific doctrine, firing a source pattern
    makes the target pattern fire in response, if they are connected
    """
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
        self.pattern = NeuralPattern(
            space_size=HyperParameters.encoder_space_size,
            value_size=HyperParameters.encoder_norm,
            generate_inplace=True
        )
        self.pattern.source_area = area
        self.pattern.source_patterns = [source, target]
        self.pattern.data = self._merge_pattern_datas(source, target)
        global GLOBAL_COUNTER
        self._id = GLOBAL_COUNTER
        GLOBAL_COUNTER += 1

    @staticmethod
    def _merge_pattern_datas(pattern1: NeuralPattern, pattern2: NeuralPattern):
        data1 = pattern1.data if isinstance(pattern1.data, dict) else {pattern1.source_area.name: pattern1.data}
        data2 = pattern2.data if isinstance(pattern2.data, dict) else {pattern2.source_area.name: pattern2.data}
        return {**data1, **data2}

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