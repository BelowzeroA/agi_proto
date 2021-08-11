# from neuro.connection import Connection
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea


class Neuron_legacy:

    def __init__(self, id: str, area: NeuralArea, container):
        self.id = id
        self.threshold = HyperParameters.default_neuron_threshold
        self.area = area
        self.container = container
        self.firing = False
        self._potential = 0
        self.history = {}
        self.initially_active = False

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, val):
        self._potential = val

    def add_to_history(self, connection: 'Connection'):
        current_tick = self.container.network.current_tick
        if current_tick not in self.history:
            self.history[current_tick] = []
        self.history[current_tick].append(connection)

    def update(self):
        self.firing = False
        current_tick = self.container.network.current_tick
        if not self.area.firing_allowed(self):
            self._potential = 0
            return

        if self.area.firing_mandatory(self):
            self.firing = True
        elif self._potential >= self.threshold or self.initially_active:
            self.firing = True

        if self.firing:
            outgoing_connections = self.container.get_outgoing_connections(self)
            for conn in outgoing_connections:
                conn.pulsing = True

        self._potential = 0

    def _repr(self):
        return f'[{self.id}]'
        return f'[{self.area.name}: {self.id}]'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()

