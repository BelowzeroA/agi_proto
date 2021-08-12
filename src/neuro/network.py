
from neuro.container import Container


class Network:

    def __init__(self, container: Container, agent: 'Agent'):
        self.container = container
        self.agent = agent
        self.current_tick = 0
        self.verbose = True
        self.container.network = self

    def step(self):
        for area in self.container.areas:
            area.update()

        for connection in self.container.connections:
            connection.update()

    def reset(self):
        for connection in self.container.connections:
            connection.pulsing = False

        for area in self.container.areas:
            area.inputs.clear()

    def run(self, max_iter=100):
        self.current_tick = 1
        while self.current_tick <= max_iter:
            self.step()

            if self.verbose:
                self.report()

            self.current_tick += 1

    def report(self):
        print(f'Tick: {self.current_tick}')
        for area in self.container.areas:
            area.report()
