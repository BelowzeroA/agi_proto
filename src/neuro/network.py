from neuro.container import NeuroContainer


class Network:

    def __init__(self, container: NeuroContainer):
        self.container = container
        self.current_tick = 0
        self.verbose = True
        self.container.network = self

    def step(self):
        for neuron in self.container.neurons:
            neuron.update()

        for connection in self.container.connections:
            connection.update_weight()

        for connection in self.container.connections:
            connection.update()

        for area in self.container.areas:
            area.update()

    def reset(self):
        for neuron in self.container.neurons:
            neuron.potential = 0

        for connection in self.container.connections:
            connection.pulsing = False

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
