from neuro.areas.encoder_area import EncoderArea
from neuro.areas.primitives_receptive_area import PrimitivesReceptiveArea
from neuro.areas.spatial_receptive_area import SpatialReceptiveArea
from neuro.hyper_params import HyperParameters
from neuro.neural_zone import NeuralZone


class VisualRecognitionZone(NeuralZone):

    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self._build_areas()

    @property
    def shape_shift_area(self):
        return self._shape_shift_area

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
            surprise_level=2,
        )

        place_presentation_area = EncoderArea.add(
            name='place representations',
            agent=self.agent,
            zone=self,
            output_space_size=HyperParameters.encoder_space_size,
            output_norm=HyperParameters.encoder_norm,
            min_inputs=2,
            surprise_level=0,
        )

        self._shape_shift_area = EncoderArea.add(
            name='shape and place',
            agent=self.agent,
            zone=self,
            output_space_size=HyperParameters.encoder_space_size,
            output_norm=HyperParameters.encoder_norm,
            min_inputs=2,
            surprise_level=0
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
            target=self.shape_shift_area
        )
        self.container.add_connection(
            source=presentation_area,
            target=self.shape_shift_area
        )

    def activate_on_body(self, body_data, prev_body_data):
        self._activate_eye_shift_areas(body_data, prev_body_data)
        self.primitives_receptive_area.activate_on_body(body_data['general_presentation'], body_data['name'])

    def _activate_eye_shift_areas(self, body_data, prev_body_data):
        from agent import ROOM_WIDTH, ROOM_HEIGHT

        half_width = ROOM_WIDTH // 2
        half_height = ROOM_HEIGHT // 2
        if prev_body_data:
            shift_x = body_data['center'][0] - prev_body_data['center'][0]
            shift_y = body_data['center'][1] - prev_body_data['center'][1]
        else:
            return
            shift_x = body_data['center'][0] - half_width
            shift_y = body_data['center'][1] - half_height
        shift_x = shift_x / half_width
        shift_y = shift_y / half_height

        eye_shift_left, eye_shift_right = (-1, shift_x) if shift_x > 0 else (-shift_x, -1)
        eye_shift_up, eye_shift_down = (-1, shift_y) if shift_y > 0 else (-shift_y, -1)

        horizontal_shift = eye_shift_left if eye_shift_left > 0 else eye_shift_right
        vertical_shift = eye_shift_down if eye_shift_down > 0 else eye_shift_up

        ratio = horizontal_shift / vertical_shift if vertical_shift > 0 else 10

        if 1.66 >= ratio >= 0.66:
            horizontal_shift_encoded = 1
            vertical_shift_encoded = 1
        elif 3 >= ratio >= 1.66:
            horizontal_shift_encoded = 1
            vertical_shift_encoded = 0.5
        elif ratio > 3:
            horizontal_shift_encoded = 1
            vertical_shift_encoded = 0
        elif 0.66 > ratio >= 0.33:
            horizontal_shift_encoded = 0.5
            vertical_shift_encoded = 1
        else:
            horizontal_shift_encoded = 0
            vertical_shift_encoded = 1

        eye_shift_left = horizontal_shift_encoded if eye_shift_left >= 0 else -1
        eye_shift_right = horizontal_shift_encoded if eye_shift_right >= 0 else -1
        eye_shift_up = vertical_shift_encoded if eye_shift_up >= 0 else -1
        eye_shift_down = vertical_shift_encoded if eye_shift_down >= 0 else -1

        self.shift_right_receptive_area.activate_on_body(eye_shift_right)
        self.shift_left_receptive_area.activate_on_body(eye_shift_left)
        self.shift_up_receptive_area.activate_on_body(eye_shift_up)
        self.shift_down_receptive_area.activate_on_body(eye_shift_down)
