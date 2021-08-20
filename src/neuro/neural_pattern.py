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

    def generate_random(self):
        self.value = random.sample(range(self.space_size), self.value_size)
        self.value.sort()

    def _repr(self):
        if self.data:
            return f'data={self.data} {self.value}'
        else:
            return f'{self.value}'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()