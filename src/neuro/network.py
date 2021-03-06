
from neuro.container import Container


class Network:
    """
    Cyclically updates the state of neural zones and areas on each step
    Also broadcasts dopamine across neural zones
    """
    def __init__(self, container: Container, agent: 'Agent'):
        self.container = container
        self.agent = agent
        self.current_tick = 0
        self.verbose = False
        self.container.network = self

    def _step_impl(self):

        for zone in self.container.zones:
            zone.on_step_begin()

        for area in self.container.areas:
            area.update()

        for connection in self.container.connections:
            connection.update()

        if self.current_tick in self.agent.dopamine_flow:
            for zone in self.container.zones:
                zone.spread_dope(self.agent.dopamine_flow[self.current_tick])

        for zone in self.container.zones:
            zone.on_step_end()

    def reset(self):
        for area in self.container.areas:
            for i in range(len(area.inputs)):
                area.inputs[i] = None

    def reset_perception(self):
        for area in self.container.areas:
            area.reset_output()

    def run(self, max_iter=100):
        """
        Main network loop
        :param max_iter: how many iterations to perform
        :return:
        """
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
