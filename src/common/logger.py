
MAX_HISTORY_SIZE = 10


class Logger:

    def __init__(self, agent):
        self.agent = agent
        self.container = agent.container
        self.network = agent.network
        self.movements = []

    def write(self):
        current_tick = self.network.current_tick
        self.movements.append((current_tick, self.agent.actions))
        if len(self.movements) > MAX_HISTORY_SIZE:
            del self.movements[0]
