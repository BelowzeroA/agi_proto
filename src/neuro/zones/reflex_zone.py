from neuro.areas.dopamine_predictor_area import DopaminePredictorArea
from neuro.areas.reflex_area import ReflexArea
from neuro.neural_zone import NeuralZone
from neuro.zones.motor_zone import MotorZone
from neuro.zones.visual_recognition_zone import VisualRecognitionZone


class ReflexZone(NeuralZone):

    def __init__(self, name: str, agent, motor_zone: MotorZone, vr_zone: VisualRecognitionZone):
        super().__init__(name, agent)
        self.motor_zone = motor_zone
        self.vr_zone = vr_zone
        self._integrate_with_motor_zone()
        self._integrate_with_visual_recognition_zone()

    def _integrate_with_motor_zone(self):
        action_areas = self.motor_zone.get_areas()
        for action_area in action_areas:

            dopamine_predictor = DopaminePredictorArea.add(
                name=f'Dope predictor: {action_area.action_id}',
                agent=self.agent,
                zone=self,
            )

            reflex_area = ReflexArea.add(
                name=f'Reflex: {action_area.action_id}',
                agent=self.agent,
                zone=self,
                action_area=action_area,
                dopamine_predictor=dopamine_predictor,
            )
            self.container.add_connection(
                source=reflex_area,
                target=action_area,
            )

    def _integrate_with_visual_recognition_zone(self):
        perceptive_area = self.vr_zone.shape_shift_area

        for area in self.areas:
            self.container.add_connection(
                source=perceptive_area,
                target=area,
            )
