from neuro.areas.hand_motion_area import HandMotionArea
from neuro.neural_zone import NeuralZone


class MotorZone(NeuralZone):

    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self._build_areas()

    def _build_areas(self):
        self._add_actions()

    def _add_actions(self):
        from agent import ACTIONS
        for action in ACTIONS:
            area = HandMotionArea.add(
                name=f'Action: {action}',
                agent=self,
                zone=self,
                action_id=action,
                output_space_size=100,
                output_norm=10
            )

