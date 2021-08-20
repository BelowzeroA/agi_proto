from neuro.areas.encoder_area import EncoderArea
from neuro.areas.primitives_receptive_area import PrimitivesReceptiveArea
from neuro.areas.spatial_receptive_area import SpatialReceptiveArea
from neuro.hyper_params import HyperParameters
from neuro.neural_zone import NeuralZone


class VisualRecognitionZone(NeuralZone):

    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self._build_areas()

    def _build_areas(self):
        self.primitives_receptive_area = PrimitivesReceptiveArea.add(
            name='primitives',
            agent=self.agent,
            zone=self,
        )

        self.shift_right_receptive_area = SpatialReceptiveArea.add(name='shift-right', agent=self.agent, zone=self)
        self.shift_left_receptive_area = SpatialReceptiveArea.add(name='shift-left', agent=self.agent, zone=self)
        self.shift_up_receptive_area = SpatialReceptiveArea.add(name='shift-up', agent=self.agent, zone=self)
        self.shift_down_receptive_area = SpatialReceptiveArea.add(name='shift-down', agent=self.agent, zone=self)

        presentation_area = EncoderArea.add(
            name='shape representations',
            agent=self.agent,
            zone=self,
            output_space_size=HyperParameters.encoder_space_size,
            output_norm=HyperParameters.encoder_norm,
        )

        place_presentation_area = EncoderArea(
            name='place representations',
            agent=self.agent,
            zone=self,
            output_space_size=HyperParameters.encoder_space_size,
            output_norm=HyperParameters.encoder_norm,
            min_inputs=2
        )

        combined_area = EncoderArea(
            name='shape and place',
            agent=self.agent,
            zone=self,
            output_space_size=HyperParameters.encoder_space_size,
            output_norm=HyperParameters.encoder_norm,
            min_inputs=2
        )

        self.container.add_connection(
            source=self.primitives_receptive_area,
            target=presentation_area
        )

        self.container.add_connection(
            source=self.shift_right_receptive_area,
            target=place_presentation_area
        )
        self.container.add_connection(
            source=self.shift_left_receptive_area,
            target=place_presentation_area
        )
        self.container.add_connection(
            source=self.shift_up_receptive_area,
            target=place_presentation_area
        )
        self.container.add_connection(
            source=self.shift_down_receptive_area,
            target=place_presentation_area
        )

        self.container.add_connection(
            source=place_presentation_area,
            target=combined_area
        )
        self.container.add_connection(
            source=presentation_area,
            target=combined_area
        )

    def activate_on_body(self, body_data, prev_body_data):
        self._activate_eye_shift_areas(body_data, prev_body_data)
        self.primitives_receptive_area.activate_on_body(body_data['general_presentation'])

    def _activate_eye_shift_areas(self, body_data, prev_body_data):
        from agent import ROOM_WIDTH, ROOM_HEIGHT

        if prev_body_data:
            shift_x = body_data['center'][0] - prev_body_data['center'][0]
            shift_y = body_data['center'][1] - prev_body_data['center'][1]
        else:

            shift_x = body_data['center'][0] - ROOM_WIDTH // 2
            shift_y = body_data['center'][1] - ROOM_HEIGHT // 2
        shift_x = shift_x / ROOM_WIDTH
        shift_y = shift_y / ROOM_HEIGHT

        eye_shift_left, eye_shift_right = (-1, shift_x) if shift_x > 0 else (-shift_x, -1)
        eye_shift_up, eye_shift_down = (-1, shift_y) if shift_y > 0 else (-shift_y, -1)

        self.shift_right_receptive_area.activate_on_body(eye_shift_right)
        self.shift_left_receptive_area.activate_on_body(eye_shift_left)
        self.shift_up_receptive_area.activate_on_body(eye_shift_up)
        self.shift_down_receptive_area.activate_on_body(eye_shift_down)