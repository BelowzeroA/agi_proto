from typing import List

from neuro.areas.reflex_area import ReflexArea
from neuro.dopamine_portion import DopaminePortion
from neuro.neural_pattern import NeuralPattern
from neuro.zones.visual_recognition_zone import AREA_NAME_VELOCITY, AREA_NAME_BODY_VELOCITY, AREA_NAME_DISTORTION, \
    AREA_NAME_SHAPE, AREA_NAME_SHAPE_SHIFT, AREA_NAME_DISTANCE, AREA_NAME_DISTANCE_CHANGE


class ReflexMoveArea(ReflexArea):
    """
    Specialized area for movement reflexes
    """
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
            zone=zone,
            action_area=action_area,
            dopamine_predictor=dopamine_predictor
        )
        self.accepted_area_names = [
            AREA_NAME_VELOCITY,
            AREA_NAME_BODY_VELOCITY,
            AREA_NAME_DISTORTION,
            AREA_NAME_SHAPE,
            AREA_NAME_SHAPE_SHIFT,
            AREA_NAME_DISTANCE,
            AREA_NAME_DISTANCE_CHANGE
        ]

    def log_inputs(self):
        if len(self.inputs) == 0:
            return
        input_lengths = []
        for input_pattern in self.inputs:
            if input_pattern is None:
                return
            input_lengths.append((input_pattern, len(input_pattern.data)))
        input_lengths.sort(key=lambda x: x[1], reverse=True)
        longest_pattern = input_lengths[0][0]
        self.agent.logger.write_content(f'Observation {longest_pattern}')

    def accepts_dopamine(self, portion: DopaminePortion) -> bool:
        return portion.source.name in self.accepted_area_names

    def receive_inputs(self, input_patterns: List[NeuralPattern]):
        inputs = []
        for pattern in input_patterns:
            if pattern.source_area:
                if pattern.source_area.name in self.accepted_area_names:
                    inputs.append(pattern)
            else:
                add = True
                for source_pattern in pattern.source_patterns:
                    if source_pattern.source_area.name not in self.accepted_area_names:
                        add = False
                if add:
                    inputs.append(pattern)
        self.inputs = inputs
