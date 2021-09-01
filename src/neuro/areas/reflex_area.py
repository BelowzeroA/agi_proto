import random
from typing import List, Union, Dict

from neuro.areas.action_area import ActionArea
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection
from neuro.sdr_processor import SDRProcessor

ACTION_LONGEVITY = 8 * HyperParameters.network_steps_per_env_step


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
        self.move_state = {}
        self.move_return = None
        self.move_id = 1
        self.caller_id = 0

    def get_connection_from(self, input_pattern: NeuralPattern):
        candidates = []
        for connection in self.connections:
            if connection.source == input_pattern:
                candidates.append((connection, connection.weight))
        if len(candidates) == 0:
            return None
        if len(candidates) == 1:
            return candidates[0][0]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def find_pattern(self, input_pattern: NeuralPattern):
        for p in self.input_patterns:
            if p == input_pattern:
                return p
        return None

    def predefined_motion(self):
        self.move_return = None
        self.caller_id = 0
        self.make_move(ticks=16, left=2)
        self.make_move(ticks=10, down=2)
        self.make_move(ticks=35, right=1)
        self.make_move(ticks=10, left=2, up=2)
        self.make_move(ticks=11, right=2, down=2)
        self.make_move(ticks=10, left=1, up=2)
        self.make_move(ticks=5, right=2)
        self.make_move(ticks=10, down=2)
        self.make_move(ticks=10, up=2)
        return self.move_return

    def make_move(self, ticks, left=0, right=0, up=0, down=0):
        self.caller_id += 1
        if self.move_return:
            return
        if self.move_id != self.caller_id:
            return

        current_tick = self.container.network.current_tick
        patterns = self.action_area.get_patterns()
        if self.move_id in self.move_state:
            if current_tick - self.move_state[self.move_id] >= ticks * HyperParameters.network_steps_per_env_step:
                self.move_return = None
                self.move_id += 1
                self.move_state[self.move_id] = current_tick
                return
        else:
            self.move_state[self.move_id] = current_tick

        if self.name == 'Reflex: move_left':
            self.move_return = patterns[left]
        elif self.name == 'Reflex: move_right':
            self.move_return = patterns[right]
        elif self.name == 'Reflex: move_up':
            self.move_return = patterns[up]
        elif self.name == 'Reflex: move_down':
            self.move_return = patterns[down]
        else:
            self.move_return = patterns[0]

    def update(self):
        current_tick = self.container.network.current_tick

        if len(self.inputs) == 1:
            combined_pattern = self.inputs[0]
        else:
            combined_pattern = SDRProcessor.make_combined_pattern(self.inputs, self.input_sizes)

        output = self.predefined_motion()
        predefined_move_chosen = output is not None

        input_pattern = None
        if combined_pattern:
            found_pattern = self.find_pattern(combined_pattern)
            if not found_pattern:
                input_pattern = combined_pattern
                self.input_patterns.append(input_pattern)
            else:
                input_pattern = found_pattern

            connection = self.get_connection_from(input_pattern)
            if connection and not predefined_move_chosen:
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
                        connection.weight = 1
                        connection.tick = causing_combination_tick
                        self.connections.append(connection)

