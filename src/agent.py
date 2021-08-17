import random

from neuro.areas.primitives_receptive_area import PrimitivesReceptiveArea
from neuro.areas.spatial_receptive_area import SpatialReceptiveArea
from neuro.container import Container
from neuro.areas.encoder_area import EncoderArea
from neuro.hyper_params import HyperParameters
from neuro.network import Network


class Agent:

    def __init__(self):
        random.seed(0)
        self.container = Container()
        self._build_network()
        self.network = Network(container=self.container, agent=self)
        self.focused_body_idx = None
        self.surprise = 0

    def _build_network(self):
        self.primitives_receptive_area = PrimitivesReceptiveArea(name='primitives', container=self.container)

        self.shift_right_receptive_area = SpatialReceptiveArea(name='shift-right', container=self.container)
        self.shift_left_receptive_area = SpatialReceptiveArea(name='shift-left', container=self.container)
        self.shift_up_receptive_area = SpatialReceptiveArea(name='shift-up', container=self.container)
        self.shift_down_receptive_area = SpatialReceptiveArea(name='shift-down', container=self.container)

        self.presentation_area = EncoderArea(
            name='shape representations',
            output_space_size=HyperParameters.encoder_space_size,
            output_activity_norm=HyperParameters.encoder_activity_norm,
            container=self.container
        )

        self.place_presentation_area = EncoderArea(
            name='place representations',
            output_space_size=HyperParameters.encoder_space_size,
            output_activity_norm=HyperParameters.encoder_activity_norm,
            container=self.container,
            min_inputs=2
        )

        self.combined_area = EncoderArea(
            name='shape and place',
            output_space_size=HyperParameters.encoder_space_size,
            output_activity_norm=HyperParameters.encoder_activity_norm,
            container=self.container,
            min_inputs=2
        )

        self.container.add_area(self.shift_right_receptive_area)
        self.container.add_area(self.shift_left_receptive_area)
        self.container.add_area(self.shift_up_receptive_area)
        self.container.add_area(self.shift_down_receptive_area)
        self.container.add_area(self.primitives_receptive_area)
        self.container.add_area(self.presentation_area)
        self.container.add_area(self.place_presentation_area)
        self.container.add_area(self.combined_area)

        self.container.add_connection(
            source=self.primitives_receptive_area,
            target=self.presentation_area
        )

        self.container.add_connection(
            source=self.shift_right_receptive_area,
            target=self.place_presentation_area
        )
        self.container.add_connection(
            source=self.shift_left_receptive_area,
            target=self.place_presentation_area
        )
        self.container.add_connection(
            source=self.shift_up_receptive_area,
            target=self.place_presentation_area
        )
        self.container.add_connection(
            source=self.shift_down_receptive_area,
            target=self.place_presentation_area
        )

        self.container.add_connection(
            source=self.place_presentation_area,
            target=self.combined_area
        )
        self.container.add_connection(
            source=self.presentation_area,
            target=self.combined_area
        )

    def on_message(self, data):
        if data == 'pattern_created':
            self.surprise += 1

    def activate_receptive_areas(self, data):
        if len(data) == 0 or len(data) > 3:
            return

        if self.focused_body_idx is None:
            self.focused_body_idx = 0
        elif self.focused_body_idx >= len(data) - 1:
            self.focused_body_idx = 0
        else:
            self.focused_body_idx += 1

        if self.container.network.verbose:
            print(f'body #{self.focused_body_idx + 1}')

        previous_focused_body_idx = self._get_previous_body_index(data)
        prev_body_data = None
        if previous_focused_body_idx:
            prev_body_data = data[previous_focused_body_idx]
        body_data = data[self.focused_body_idx]
        self._serial_activate_on_body(body_data, prev_body_data)

    def _get_previous_body_index(self, data):
        if len(data) < 2:
            return None
        if self.focused_body_idx == 0:
            return len(data) - 1
        else:
            return self.focused_body_idx - 1

    def _serial_activate_on_body(self, body_data, prev_body_data):
        room_width = 640
        room_height = 480
        if prev_body_data:
            shift_x = body_data['center'][0] - prev_body_data['center'][0]
            shift_y = body_data['center'][1] - prev_body_data['center'][1]
        else:
            shift_x = body_data['center'][0] - room_width // 2
            shift_y = body_data['center'][1] - room_height // 2
        shift_x = shift_x / room_width
        shift_y = shift_y / room_height

        eye_shift_left, eye_shift_right = (-1, shift_x) if shift_x > 0 else (-shift_x, -1)
        eye_shift_up, eye_shift_down = (-1, shift_y) if shift_y > 0 else (-shift_y, -1)

        self.primitives_receptive_area.activate_on_body(body_data['general_presentation'])

        self.shift_right_receptive_area.activate_on_body(eye_shift_right)
        self.shift_left_receptive_area.activate_on_body(eye_shift_left)
        self.shift_up_receptive_area.activate_on_body(eye_shift_up)
        self.shift_down_receptive_area.activate_on_body(eye_shift_down)

        self.surprise = 0

        self.network.verbose = False
        self.network.step()
        # self.network.reset()

    def env_step(self, data):
        self.activate_receptive_areas(data)
        if self.container.network.verbose:
            print(f'Surprise: {self.surprise}')

