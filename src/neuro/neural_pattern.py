import random


class NeuralPattern:

    def __init__(self, space_size: int, value_size: int = 0, value=None, data=None):
        if value:
            self.value = value
            self.value_size = len(value)
        else:
            self.value = []
            self.value_size = value_size
        self.data = data
        self.space_size = space_size
        self.history = {}

    def generate_random(self):
        self.value = random.sample(range(self.space_size), self.value_size)
        self.value.sort()

    def log(self, area: 'NeuralArea'):
        current_tick = area.container.network.current_tick
        self.history[current_tick] = area

    def merge_histories(self, histories: list):
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


    def _repr(self):
        if self.data is not None:
            return f'data={self.data} {self.value}'
        else:
            return f'{self.value}'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()