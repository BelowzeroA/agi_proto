from neuro.areas.dopamine_anticipator_area import DopamineAnticipatorArea
from neuro.areas.dopamine_predictor_area import DopaminePredictorArea
from neuro.areas.reflex_area import ReflexArea
from neuro.areas.reflex_grab_area import ReflexGrabArea
from neuro.areas.reflex_move_area import ReflexMoveArea
from neuro.neural_zone import NeuralZone
from neuro.patterns_combiner import PatternsCombiner
from neuro.zones.motor_zone import MotorZone
from neuro.zones.tactile_zone import TactileZone
from neuro.zones.visual_recognition_zone import VisualRecognitionZone


class ReflexZone(NeuralZone):
    """
    Manages the reflex areas and the areas related to the dopamine flow system
    """
    def __init__(self, name: str, agent, motor_zone: MotorZone, vr_zone: VisualRecognitionZone, ta_zone: TactileZone):
        super().__init__(name, agent)
        self.accumulated_dope = 0
        self.motor_zone = motor_zone
        self.vr_zone = vr_zone
        self.ta_zone = ta_zone
        self.combiner = PatternsCombiner(name=f'Combiner: {name}', agent=agent, zone=self)
        self._integrate_with_motor_zone()
        self._integrate_with_perception()

    def _integrate_with_motor_zone(self):
        action_areas = self.motor_zone.get_areas()
        for action_area in action_areas:

            dopamine_predictor = DopaminePredictorArea.add(
                name=f'Dope predictor: {action_area.action_id}',
                agent=self.agent,
                zone=self,
            )

            if action_area.action_id == 'move':
                reflex_area = ReflexMoveArea.add(
                    name=f'Reflex: {action_area.action_id}',
                    agent=self.agent,
                    zone=self,
                    action_area=action_area,
                    dopamine_predictor=dopamine_predictor,
                )
            elif action_area.action_id == 'grab':
                reflex_area = ReflexGrabArea.add(
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

    def _integrate_with_perception(self):

        self.container.add_connection(
            source=self.ta_zone.output_area,
            target=self.combiner,
        )

        self.container.add_connection(
            source=self.vr_zone.shape_shift_area,
            target=self.combiner,
        )

        self.container.add_connection(
            source=self.vr_zone.distance,
            target=self.combiner,
        )

        self.container.add_connection(
            source=self.vr_zone.distance_change,
            target=self.combiner,
        )

        for area in self.areas:
            if isinstance(area, ReflexArea):
                self.combiner.output_areas.append(area)

        dopamine_anticipator = DopamineAnticipatorArea.add(
            name=f'dope anticipator',
            agent=self.agent,
            zone=self,
        )

        self.combiner.output_areas.append(dopamine_anticipator)

    def receive_self_induced_dope(self, dope_value: int):
        self.accumulated_dope += dope_value

    def on_step_begin(self):
        self.accumulated_dope = 0
        self.combiner.combine_transfer()

    def on_step_end(self):
        # Not implemented for dopamine portions mode
        return
        # Spread dopamine-induced excitation across all the reflex areas
        if self.accumulated_dope:
            for area in self.areas:
                # if isinstance(area, ReflexArea):
                area.receive_dope(self.accumulated_dope, self_induced=True)
