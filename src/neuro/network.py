
from neuro.container import Container


class Network:

    def __init__(self, container: Container, agent: 'Agent'):
        self.container = container
        self.agent = agent
        self.current_tick = 0
        self.verbose = False
        self.container.network = self

    def _step_impl(self):
        for area in self.container.areas:
            area.update()

        for connection in self.container.connections:
            connection.update()

        if self.agent.surprise:
            for zone in self.container.zones:
                zone.spread_dope(self.agent.surprise)

    def reset(self):
        for area in self.container.areas:
            area.inputs.clear()

    def run(self, max_iter=100):
        self.current_tick = 1
        while self.current_tick <= max_iter:
            self._step_impl()

            if self.verbose:
                self.report()

            self.current_tick += 1

    def step(self):
        self.current_tick += 1
        self._step_impl()

        if self.verbose:
            self.report()

    def report(self):
        print(f'Tick: {self.current_tick}')
        for area in self.container.areas:
            area.report()
