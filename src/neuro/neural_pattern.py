import random

GLOBAL_COUNTER = 0
all_patterns = []


class NeuralPattern:

    def __init__(self, space_size: int, value_size: int = 0, value=None, data=None):
        global GLOBAL_COUNTER, all_patterns
        if value:
            self.value = value
            self.value_size = len(value)
        else:
            self.value = []
            self.value_size = value_size
        self.data = data
        self.source_patterns = []
        self.space_size = space_size
        self.history = {}
        self._id = GLOBAL_COUNTER
        GLOBAL_COUNTER += 1
        all_patterns.append(self)

    @classmethod
    def find_or_create(cls, space_size: int, value_size: int = 0, value=None, data=None):
        global all_patterns

        if value:
            for pattern in all_patterns:
                if space_size != pattern.space_size or len(value) != pattern.value_size:
                    continue
                intersection = set(value) & set(pattern.value)
                if len(intersection) == len(value):
                    return pattern
        return cls(space_size=space_size, value_size=value_size, value=value, data=data)

    def __eq__(self, other):
        if self.value_size != other.value_size or self.space_size != other.space_size:
            return False
        intersection = set(self.value) & set(other.value)
        return len(intersection) == self.value_size

    def similarity(self, other):
        if self.value_size != other.value_size or self.space_size != other.space_size:
            return 0
        intersection = set(self.value) & set(other.value)
        return len(intersection) / self.value_size

    def generate_random(self):
        self.value = random.sample(range(self.space_size), self.value_size)
        self.value.sort()

    def log(self, area: 'NeuralArea'):
        current_tick = area.container.network.current_tick
        self.history[current_tick] = [area]

    def merge_histories(self, histories: list):
        return
        all_ticks = set()
        for history in histories:
            for tick in history:
                all_ticks.add(tick)
        all_ticks = sorted(list(all_ticks))

        for tick in all_ticks:
            for history in histories:
                if tick in history:
                    if tick not in self.history:
                        self.history[tick] = []
                    self.history[tick].append(history[tick])

    def _repr0(self):
        if self.data is not None:
            return f'({self._id}) data={self.data} {self.value}'
        else:
            return f'({self._id}) {self.value}'

    def _repr(self):
        if self.data is not None:
            return f'({self._id}) {self.data}'
        else:
            return f'({self._id})'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()