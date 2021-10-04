import random
from typing import List, Union, Dict

import pattern as pattern

from neuro.areas.action_area import ActionArea
from neuro.areas.dopamine_predictor_area import DopaminePredictorArea
from neuro.dopamine_portion import DopaminePortion
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection

ACTION_LONGEVITY = 8 * HyperParameters.network_steps_per_env_step


class ReflexArea(NeuralArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
            action_area,
            dopamine_predictor,
    ):
        super().__init__(
            name=name,
            agent=agent,
            zone=zone
        )
        self.input_patterns = []
        self.action_area: ActionArea = action_area
        self.dopamine_predictor: DopaminePredictorArea = dopamine_predictor
        self.pattern_start_tick = None
        self.connections: List[PatternsConnection] = []
        self.active_pattern = None
        self.history = {}
        self.move_state = {}
        self.move_return = None
        self.move_id = 1
        self.caller_id = 0

    def get_connection_from(self, input_patterns: List[NeuralPattern]):
        current_tick = self.agent.network.current_tick
        candidates = []
        for connection in self.connections:
            for pattern in input_patterns:
                if connection.source == pattern and current_tick - connection.tick > 30:
                    candidates.append((connection, connection.weight))
        if len(candidates) == 0:
            return None

        if len(candidates) > 1:
            candidates.sort(key=lambda x: x[1], reverse=True)
        top_weight = candidates[0][1]
        if top_weight < 0.15:
            return None

        top_connections = [c for c in candidates if c[1] == top_weight]
        if len(top_connections) > 1:
            top_connection = random.choice(top_connections)
        else:
            top_connection = top_connections[0]
        return top_connection[0]

    def get_connection_from_to(self, source: NeuralPattern, target: NeuralPattern) -> PatternsConnection:
        for connection in self.connections:
            if connection.source == source and connection.target == target:
                return connection
        return None

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
        self.make_move(ticks=10, left=1, up=2, grab=0)
        self.make_move(ticks=5, right=2, grab=0)
        self.make_move(ticks=10, down=2)
        self.make_move(ticks=10, up=2)
        self.make_move(ticks=10, right=2)
        self.make_move(ticks=10, left=2, down=2)
        self.make_move(ticks=15, right=2, up=1)
        self.make_move(ticks=10, down=2)
        self.make_move(ticks=13, left=2)
        self.make_move(ticks=20, left=0, up=2, grab=1)
        return self.move_return

    def predefined_motion0(self):
        self.move_return = None
        self.caller_id = 0
        self.make_move(ticks=16, left=2)
        self.make_move(ticks=10, down=2)
        self.make_move(ticks=35, right=1)
        self.make_move(ticks=25, up=2, grab=1)
        self.make_move(ticks=20, right=2, grab=1)
        self.make_move(ticks=5, down=2, grab=1)
        self.make_move(ticks=5, up=2, grab=0)
        self.make_move(ticks=20, left=2, grab=0)
        return self.move_return

    def make_move(self, ticks, left=0, right=0, up=0, down=0, grab=0):
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

        if self.name == 'Reflex: move':
            for pattern in patterns:
                if pattern.data['left'] == left and pattern.data['right'] == right and \
                        pattern.data['down'] == down and pattern.data['up'] == up:
                    self.move_return = pattern
                    break
        elif self.name == 'Reflex: grab':
            self.move_return = patterns[grab]
        else:
            self.move_return = patterns[0]

    def log_inputs(self):
        pass

    def receive_inputs(self, input_patterns: List[NeuralPattern]):
        pass

    def update(self):
        current_tick = self.container.network.current_tick

        self.log_inputs()

        output = self.predefined_motion()
        # output = None
        predefined_move_chosen = output is not None

        if not predefined_move_chosen:
            if self.active_pattern is None or current_tick - self.pattern_start_tick >= ACTION_LONGEVITY:
                connection = self.get_connection_from(self.inputs)
                if connection:
                    self.dopamine_predictor.on_connection_activated(connection)
                    self.pattern_start_tick = current_tick
                    output = connection.target
                    self.active_pattern = output
            elif self.active_pattern:
                output = self.active_pattern

        if not output:
            # Choosing a random action
            patterns = self.action_area.get_patterns()
            if self.pattern_start_tick is None or current_tick - self.pattern_start_tick > ACTION_LONGEVITY:
                self.pattern_start_tick = current_tick
                self.active_pattern = random.choice(patterns)
            output = self.active_pattern

        self.history[current_tick] = (self.inputs, output)
        # output.log(self)
        self.output = output

    def _process_input_output_combination(
            self,
            combination_tick: int,
            dopamine_portions: List[DopaminePortion],
            weight: float,
            processed_connections: set
    ):
        combination = self.history[combination_tick]
        input_patterns = combination[0]
        alive_patterns = [p for p in input_patterns if p]
        if len(alive_patterns) == 0:
            return

        dope_value = sum(portion.value for portion in dopamine_portions)

        for input_pattern in alive_patterns:
            connection = self.get_connection_from_to(source=input_pattern, target=combination[1])
            if not connection:
                connection = PatternsConnection.add(
                    source=input_pattern,
                    target=combination[1],
                    agent=self.agent,
                    weight=weight,
                    tick=combination_tick,
                    dope_value=dope_value,
                    area=self
                )
                self.connections.append(connection)
                processed_connections.add(connection)
                self.agent.logger.write_content(f'connection created {connection}, weight={connection.weight}')
            else:
                if connection not in processed_connections:
                    processed_connections.add(connection)
                    connection.update_weight(dope_value * HyperParameters.learning_rate)
                else:
                    connection.weight = weight

    def accepts_dopamine(self, portion: DopaminePortion) -> bool:
        return False

    def receive_dope(self, dopamine_portions: List[DopaminePortion], self_induced=False):
        current_tick = self.container.network.current_tick
        selected_portions = [portion for portion in dopamine_portions if self.accepts_dopamine(portion)]

        if len(selected_portions) == 0:
            return

        processed_connections = set()
        start_tick = current_tick - 4 if self_induced else current_tick - 8
        for causing_combination_tick in range(start_tick, current_tick - 2):
            if causing_combination_tick in self.history:
                weight = 0.1 * (causing_combination_tick - start_tick + 1) + 0.2
                self._process_input_output_combination(
                    causing_combination_tick,
                    selected_portions,
                    weight,
                    processed_connections
                )
