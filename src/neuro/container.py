from typing import List

from neuro.inter_area_connection import InterAreaConnection
from neuro.neural_area import NeuralArea


class Container:

    def __init__(self):
        self.network = None
        self.areas = []
        self.connections = []

    def add_area(self, area):
        self.areas.append(area)

    def get_outgoing_connections(self, area: NeuralArea) -> List[InterAreaConnection]:
        return [c for c in self.connections if c.source == area]

    def add_connection(self, source: NeuralArea, target: NeuralArea) -> InterAreaConnection:
        conn = InterAreaConnection(source=source, target=target)
        self.connections.append(conn)
        conn.on_adding()
        return conn