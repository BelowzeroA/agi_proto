import random
from typing import List, Union, Dict

from neuro.areas.action_area import ActionArea
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern

ACTION_SPACE = [0, 1, 2]
ACTION_LONGEVITY = 8


class HandMotionArea(ActionArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
            action_id: str,
            output_space_size: int,
            output_norm: int,
    ):
        super().__init__(
            name=name,
            agent=agent,
            zone=zone,
            action_id=action_id,
            output_space_size=output_space_size,
            output_norm=output_norm,
        )
        self.patterns: List[NeuralPattern] = []
        self._generate_action_patterns()
        self.pattern_start_tick = None
        self.action_value = 0

    def get_patterns(self):
        return self.patterns

    def _generate_action_patterns(self):
        for i in ACTION_SPACE:
            if self.name == 'Action: grab' and i == 2:
                continue
            pattern = NeuralPattern(space_size=self.output_space_size, value_size=self.output_norm)
            pattern.generate_random()
            pattern.data = i
            self.patterns.append(pattern)

    def update(self):
        input_pattern = self.inputs[0]
        if not input_pattern:
            return

        self.agent.on_message({
            'message': 'hand_move',
            'action_id': self.action_id,
            'action_value': input_pattern.data
        })
        self.output = input_pattern
        # self.output = self.patterns[self.action_value]
