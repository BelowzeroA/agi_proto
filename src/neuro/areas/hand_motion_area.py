import random
from typing import List, Union, Dict

from neuro.areas.action_area import ActionArea
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern

GRAB_ACTION_SPACE = [0, 1]
ACTION_LONGEVITY = 8


class HandMotionArea(ActionArea):
    """
    Outputs the action to an agent. Actions can be either movements or palm clenching
    """
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

    def _get_motion_patterns(self):
        actions = []
        actions.append({'left': 0, 'right': 0, 'up': 0, 'down': 0})

        actions.append({'left': 1, 'right': 0, 'up': 0, 'down': 0})
        actions.append({'left': 2, 'right': 0, 'up': 0, 'down': 0})
        actions.append({'left': 0, 'right': 1, 'up': 0, 'down': 0})
        actions.append({'left': 0, 'right': 2, 'up': 0, 'down': 0})

        actions.append({'left': 0, 'right': 0, 'up': 1, 'down': 0})
        actions.append({'left': 0, 'right': 0, 'up': 2, 'down': 0})
        actions.append({'left': 0, 'right': 0, 'up': 0, 'down': 1})
        actions.append({'left': 0, 'right': 0, 'up': 0, 'down': 2})

        actions.append({'left': 1, 'right': 0, 'up': 1, 'down': 0})
        actions.append({'left': 2, 'right': 0, 'up': 1, 'down': 0})
        actions.append({'left': 1, 'right': 0, 'up': 2, 'down': 0})
        actions.append({'left': 2, 'right': 0, 'up': 2, 'down': 0})

        actions.append({'left': 0, 'right': 1, 'up': 1, 'down': 0})
        actions.append({'left': 0, 'right': 2, 'up': 1, 'down': 0})
        actions.append({'left': 0, 'right': 1, 'up': 2, 'down': 0})
        actions.append({'left': 0, 'right': 2, 'up': 2, 'down': 0})

        actions.append({'left': 1, 'right': 0, 'up': 0, 'down': 1})
        actions.append({'left': 2, 'right': 0, 'up': 0, 'down': 1})
        actions.append({'left': 1, 'right': 0, 'up': 0, 'down': 2})
        actions.append({'left': 2, 'right': 0, 'up': 0, 'down': 2})

        actions.append({'left': 0, 'right': 1, 'up': 0, 'down': 1})
        actions.append({'left': 0, 'right': 2, 'up': 0, 'down': 1})
        actions.append({'left': 0, 'right': 1, 'up': 0, 'down': 2})
        actions.append({'left': 0, 'right': 2, 'up': 0, 'down': 2})
        return actions

    def _generate_action_patterns(self):
        if self.name == 'Action: move':
            for data in self._get_motion_patterns():
                pattern = NeuralPattern(
                    space_size=self.output_space_size,
                    value_size=self.output_norm,
                    source_area=self
                )
                pattern.generate_random()
                pattern.data = data
                self.patterns.append(pattern)

        elif self.name == 'Action: grab':
            for data in GRAB_ACTION_SPACE:
                pattern = NeuralPattern(
                    space_size=self.output_space_size,
                    value_size=self.output_norm,
                    source_area=self,
                )
                pattern.generate_random()
                pattern.data = data
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
