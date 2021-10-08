from typing import List

from neuro.inter_area_connection import InterAreaConnection
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.neural_zone import NeuralZone
from neuro.patterns_connection import PatternsConnection


class Container:
    """
    Contains neural zones, areas and connections between areas i.e. connectome
    """
    def __init__(self):
        self.network = None
        self.areas = []
        self.connections = []
        self.zones = []
        self.pattern_connections = []

    def add_area(self, area):
        self.areas.append(area)

    def add_zone(self, zone: NeuralZone):
        self.zones.append(zone)

    def get_outgoing_connections(self, area: NeuralArea) -> List[InterAreaConnection]:
        return [c for c in self.connections if c.source == area]

    def add_connection(self, source: NeuralArea, target: NeuralPattern) -> InterAreaConnection:
        connections = [c for c in self.connections if c.source == source and c.target == target]
        assert len(connections) == 0, f'Connection between {source} and {target} already exists'
        conn = InterAreaConnection(source=source, target=target)
        self.connections.append(conn)
        conn.on_adding()
        return conn

    def add_patterns_connection(self, connection: PatternsConnection) -> None:
        connections = [c for c in self.pattern_connections if
                       c.source == connection.source and c.target == connection.target]
        assert len(connections) == 0, f'Connection between {connection.source} and {connection.target} already exists'
        self.pattern_connections.append(connection)

    def get_area_by_name(self, name: str):
        selected = [area for area in self.areas if area.name == name]
        if len(selected) == 0:
            raise AttributeError(f'No area named "{name}"')
        return selected[0]
