
class NeuralZone:
    """
    Abstract class for storing neural areas and implementing some cognitive function
    """
    def __init__(self, name: str, agent):
        self.container = agent.container
        self.areas = []
        self.name = name
        self.agent = agent

    @classmethod
    def add(cls, name, agent) -> 'NeuralZone':
        zone = cls(name, agent)
        agent.container.add_zone(zone)
        return zone

    def on_area_updated(self, area):
        pass

    def report(self):
        pass

    def _repr(self):
        return f'{self.name}'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()


