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
        self.patterns: Dict[int, NeuralPattern] = {}
        self._generate_action_patterns()
        self.pattern_start_tick = None
        self.action_value = 0

    def _generate_action_patterns(self):
        for i in ACTION_SPACE:
            pattern = NeuralPattern(space_size=self.output_space_size, value_size=self.output_norm)
            pattern.generate_random()
            self.patterns[i] = pattern

    def update(self):
        current_tick = self.container.network.current_tick
        if self.pattern_start_tick is None or current_tick - self.pattern_start_tick > ACTION_LONGEVITY:
            self.pattern_start_tick = current_tick
            self.action_value = random.choice(ACTION_SPACE)
        agent = self.container.network.agent
        agent.on_message({
            'message': 'hand_move',
            'action_id': self.action_id,
            'action_value': self.action_value
        })
        self.output = self.patterns[self.action_value]
