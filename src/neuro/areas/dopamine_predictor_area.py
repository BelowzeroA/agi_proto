from typing import List, Dict

from neuro.dopamine_portion import DopaminePortion
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection


TRACE_INTERVAL = 20 * HyperParameters.network_steps_per_env_step


class TracedConnection:
    def __init__(self, connection: PatternsConnection, start_tick: int):
        self.connection = connection
        self.start_tick = start_tick
        self.control_tick = start_tick + TRACE_INTERVAL
        self.updated = False
        self.accumulated_dope = 0


class DopaminePredictorArea(NeuralArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
    ):
        super().__init__(name=name, agent=agent, zone=zone)
        self.traced_connections: List[TracedConnection] = []

    def update(self):
        current_tick = self.agent.network.current_tick
        for tc in self.traced_connections:
            if tc.updated:
                continue
            if current_tick > tc.control_tick:
                difference = tc.accumulated_dope - tc.connection.dope_value
                tc.connection.update_weight(difference * HyperParameters.learning_rate)
                tc.updated = True

    def on_connection_activated(self, connection: PatternsConnection):
        current_tick = self.agent.network.current_tick
        traced_connection = TracedConnection(
            connection=connection,
            start_tick=current_tick,
        )
        self.traced_connections.append(traced_connection)
        self.agent.logger.write_content(f'starting to trace {connection}')

    def receive_dope(self, dopamine_portions: List[DopaminePortion], self_induced=False):
        # if dope_value < 2:
        #     return
        current_tick = self.container.network.current_tick
        for tc in self.traced_connections:
            if tc.updated:
                continue
            for portion in dopamine_portions:
                if tc.start_tick < current_tick <= tc.control_tick and tc.connection.area.accepts_dopamine(portion):
                    tc.accumulated_dope += portion.value



