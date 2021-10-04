from typing import List

from neuro.areas.reflex_area import ReflexArea
from neuro.dopamine_portion import DopaminePortion
from neuro.neural_pattern import NeuralPattern
from neuro.zones.tactile_zone import AREA_NAME_TOUCH
from neuro.zones.visual_recognition_zone import AREA_NAME_DISTANCE


class ReflexGrabArea(ReflexArea):

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
        self.accepted_area_names = [AREA_NAME_TOUCH, AREA_NAME_DISTANCE]

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
