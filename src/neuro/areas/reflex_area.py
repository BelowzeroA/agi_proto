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
        self.input_patterns = []
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

    def find_pattern(self, input_pattern: NeuralPattern):
        for p in self.input_patterns:
            if p == input_pattern:
                return p
        return None

    def predefined_motion(self):
        current_tick = self.container.network.current_tick

        if current_tick < 20:
            return self.make_move(left=2)
        elif current_tick < 30:
            return self.make_move(down=2)
        elif current_tick < 72:
            return self.make_move(right=1)
        elif current_tick < 78:
            return self.make_move(left=2, up=2)
        elif current_tick < 90:
            return self.make_move(right=1, down=1)
        elif current_tick < 96:
            return self.make_move(left=2, up=2)
        return None

    def make_move(self, left=0, right=0, up=0, down=0):
        patterns = self.action_area.get_patterns()
        if self.name == 'Reflex: move_left':
            return patterns[left]
        elif self.name == 'Reflex: move_right':
            return patterns[right]
        elif self.name == 'Reflex: move_up':
            return patterns[up]
        elif self.name == 'Reflex: move_down':
            return patterns[down]
        else:
            return patterns[0]

    def update(self):
        current_tick = self.container.network.current_tick

        if len(self.inputs) == 1:
            combined_pattern = self.inputs[0]
        else:
            combined_pattern = SDRProcessor.make_combined_pattern(self.inputs, self.input_sizes)

        output = self.predefined_motion()

        input_pattern = None
        if combined_pattern:
            found_pattern = self.find_pattern(combined_pattern)
            if not found_pattern:
                input_pattern = combined_pattern
                self.input_patterns.append(input_pattern)
            else:
                input_pattern = found_pattern

            connection = self.get_connection_from(input_pattern)
            if connection:
                output = connection.target

        if not output:
            patterns = self.action_area.get_patterns()
            if self.pattern_start_tick is None or current_tick - self.pattern_start_tick > ACTION_LONGEVITY:
                self.pattern_start_tick = current_tick
                self.active_pattern = random.choice(patterns)
            output = self.active_pattern

        self.history[current_tick] = (input_pattern, output)
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
                        connection.tick = causing_combination_tick
                        self.connections.append(connection)

