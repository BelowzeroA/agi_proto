import random
from typing import List, Union, Dict

from neuro.areas.action_area import ActionArea
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection
from neuro.sdr_processor import SDRProcessor

ACTION_LONGEVITY = 8


class ReflexArea(NeuralArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
            action_area,
    ):
        super().__init__(
            name=name,
            agent=agent,
            zone=zone
        )
        self.action_area: ActionArea = action_area
        self.pattern_start_tick = None
        self.connections: List[PatternsConnection] = []
        self.active_pattern = None
        self.history = {}

    def get_connection_from(self, input_pattern: NeuralPattern):
        for connection in self.connections:
            if connection.source == input_pattern:
                return connection
        return None

    def predefined_motion(self):
        current_tick = self.container.network.current_tick
        patterns = self.action_area.get_patterns()
        if current_tick < 25:
            if self.name == 'Reflex: move_left':
                return patterns[2]
            else:
                return patterns[0]
        elif current_tick < 40:
            if self.name == 'Reflex: move_down':
                return patterns[2]
            else:
                return patterns[0]

        elif current_tick < 100:
            if self.name == 'Reflex: move_right':
                return patterns[1]
            else:
                return patterns[0]


    def update(self):
        current_tick = self.container.network.current_tick

        combined_pattern = SDRProcessor.make_combined_pattern(self.inputs, self.input_sizes)
        output = None

        if current_tick < 100:
            output = self.predefined_motion()

        if combined_pattern:
            connection = self.get_connection_from(combined_pattern)
            if connection:
                output = connection.target

        if not output:
            patterns = self.action_area.get_patterns()
            if self.pattern_start_tick is None or current_tick - self.pattern_start_tick > ACTION_LONGEVITY:
                self.pattern_start_tick = current_tick
                self.active_pattern = random.choice(patterns)
            output = self.active_pattern

        self.history[current_tick] = (combined_pattern, output)
        output.log(self)

        self.output = output

    def receive_dope(self, dope_value: int):
        current_tick = self.container.network.current_tick
        if dope_value > 1:
            causing_combination_tick = current_tick - 2
            if causing_combination_tick in self.history:
                combination = self.history[causing_combination_tick]
                input_pattern = combination[0]
                if input_pattern:
                    connection = self.get_connection_from(input_pattern)
                    if not connection:
                        connection = PatternsConnection(
                            source=input_pattern,
                            target=combination[1]
                        )
                        self.connections.append(connection)

