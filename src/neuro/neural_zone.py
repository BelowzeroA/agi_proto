
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
    def add(cls, name, agent, **kwargs) -> 'NeuralZone':
        zone = cls(name, agent, **kwargs)
        agent.container.add_zone(zone)
        return zone

    def get_areas(self):
        return self.areas

    def spread_dope(self, dope_value: int):
        for area in self.areas:
            area.receive_dope(dope_value, self_induced=False)

    def on_area_updated(self, area):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass

    def report(self):
        pass

    def _repr(self):
        return f'{self.name}'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()


