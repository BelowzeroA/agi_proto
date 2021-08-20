from typing import List

from neuro.inter_area_connection import InterAreaConnection
from neuro.neural_area import NeuralArea
from neuro.neural_zone import NeuralZone


class Container:

    def __init__(self):
        self.network = None
        self.areas = []
        self.connections = []
        self.zones = []

    def add_area(self, area):
        self.areas.append(area)

    def add_zone(self, zone: NeuralZone):
        self.zones.append(zone)

    def get_outgoing_connections(self, area: NeuralArea) -> List[InterAreaConnection]:
        return [c for c in self.connections if c.source == area]

    def add_connection(self, source: NeuralArea, target: NeuralArea) -> InterAreaConnection:
        connections = [c for c in self.connections if c.source == source and c.target == target]
        assert len(connections) == 0, f'Connection between {source} and {target} already exists'
        conn = InterAreaConnection(source=source, target=target)
        self.connections.append(conn)
        conn.on_adding()
        return conn