from neuro.areas.dopamine_anticipator_area import DopamineAnticipatorArea
from neuro.areas.dopamine_predictor_area import DopaminePredictorArea
from neuro.areas.reflex_area import ReflexArea
from neuro.neural_zone import NeuralZone
from neuro.zones.motor_zone import MotorZone
from neuro.zones.visual_recognition_zone import VisualRecognitionZone


class ReflexZone(NeuralZone):

    def __init__(self, name: str, agent, motor_zone: MotorZone, vr_zone: VisualRecognitionZone):
        super().__init__(name, agent)
        self.accumulated_dope = 0
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
            if isinstance(area, ReflexArea):
                self.container.add_connection(
                    source=perceptive_area,
                    target=area,
                )

        dopamine_anticipator = DopamineAnticipatorArea.add(
            name=f'dope anticipator',
            agent=self.agent,
            zone=self,
        )

        self.container.add_connection(
            source=perceptive_area,
            target=dopamine_anticipator,
        )

    def receive_self_induced_dope(self, dope_value: int):
        self.accumulated_dope += dope_value

    def on_step_begin(self):
        self.accumulated_dope = 0

    def on_step_end(self):
        # Spread dopamine-induced excitation across all the reflex areas
        if self.accumulated_dope:
            for area in self.areas:
                # if isinstance(area, ReflexArea):
                area.receive_dope(self.accumulated_dope, self_induced=True)
