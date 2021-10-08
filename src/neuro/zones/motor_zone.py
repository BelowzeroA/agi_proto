from neuro.areas.hand_motion_area import HandMotionArea
from neuro.neural_zone import NeuralZone


class MotorZone(NeuralZone):
    """
    Inventory of all the actions of an agent. Actions can be either movements or palm clenching
    """
    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self._build_areas()

    def _build_areas(self):
        self._add_actions()

    def _add_actions(self):
        from agent import MACRO_ACTIONS
        for action in MACRO_ACTIONS:
            area = HandMotionArea.add(
                name=f'Action: {action}',
                agent=self.agent,
                zone=self,
                action_id=action,
                output_space_size=100,
                output_norm=10
            )

