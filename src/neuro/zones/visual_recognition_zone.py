import math

from neuro.areas.body_shape_distortion_area import BodyShapeDistortionArea
from neuro.areas.encoder_area import EncoderArea
from neuro.areas.primitives_receptive_area import PrimitivesReceptiveArea
from neuro.areas.spatial_receptive_area import SpatialReceptiveArea
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_zone import NeuralZone

AREA_NAME_VELOCITY = 'velocity'
AREA_NAME_BODY_VELOCITY = 'body velocity'
AREA_NAME_DISTORTION = 'distortion'
AREA_NAME_SHAPE = 'shape'
AREA_NAME_SHAPE_SHIFT = 'shape and shift'
AREA_NAME_DISTANCE = 'distance'
AREA_NAME_DISTANCE_CHANGE = 'distance change'


class VisualRecognitionZone(NeuralZone):
    """
    Manages the areas related to visual perception
    """
    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self.distance_to_bodies = {}
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

        self.shape = EncoderArea.add(
            name=AREA_NAME_SHAPE,
            agent=self.agent,
            zone=self,
            surprise_level=2,
            recognition_threshold=0.9
        )

        place_presentation_area = EncoderArea.add(
            name='shift',
            agent=self.agent,
            zone=self,
            min_inputs=2,
            surprise_level=0,
            convey_new_pattern=True,
        )

        self._shape_shift_area = EncoderArea.add(
            name=AREA_NAME_SHAPE_SHIFT,
            agent=self.agent,
            zone=self,
            min_inputs=2,
            surprise_level=0,
            recognition_threshold=0.9,
            convey_new_pattern=True,
            cached_output_num_ticks=15,
            accepts_dopamine_from=[AREA_NAME_DISTORTION]
        )

        self.container.add_connection(source=self.primitives_receptive_area, target=self.shape)

        self.container.add_connection(source=self.shift_right_receptive_area, target=place_presentation_area)
        self.container.add_connection(source=self.shift_left_receptive_area, target=place_presentation_area)
        self.container.add_connection(source=self.shift_up_receptive_area, target=place_presentation_area)
        self.container.add_connection(source=self.shift_down_receptive_area, target=place_presentation_area)

        self.container.add_connection(source=place_presentation_area, target=self.shape_shift_area)
        self.container.add_connection(source=self.shape, target=self.shape_shift_area)

        self.body_distortion = BodyShapeDistortionArea.add(name=AREA_NAME_DISTORTION, agent=self.agent, zone=self)

        self.velocity_left = SpatialReceptiveArea.add(name='velocity-left', agent=self.agent, zone=self, grid_size=10)
        self.velocity_right = SpatialReceptiveArea.add(name='velocity-right', agent=self.agent, zone=self, grid_size=10)
        self.velocity_up = SpatialReceptiveArea.add(name='velocity-up', agent=self.agent, zone=self, grid_size=10)
        self.velocity_down = SpatialReceptiveArea.add(name='velocity-down', agent=self.agent, zone=self, grid_size=10)

        self.velocity = EncoderArea.add(
            name=AREA_NAME_VELOCITY,
            agent=self.agent,
            zone=self,
            min_inputs=1,
            convey_new_pattern=True,
            surprise_level=0,
            recognition_threshold=0.9,
        )

        self.body_velocity = EncoderArea.add(
            name=AREA_NAME_BODY_VELOCITY,
            agent=self.agent,
            zone=self,
            min_inputs=2,
            surprise_level=2,
            recognition_threshold=0.9,
        )

        self.container.add_connection(source=self.velocity_left, target=self.velocity)
        self.container.add_connection(source=self.velocity_right, target=self.velocity)
        self.container.add_connection(source=self.velocity_up, target=self.velocity)
        self.container.add_connection(source=self.velocity_down, target=self.velocity)

        self.container.add_connection(source=self.velocity, target=self.body_velocity)
        self.container.add_connection(source=self.shape, target=self.body_velocity)

        self.distance_receptive_area = SpatialReceptiveArea.add(
            name='distance-receptive',
            agent=self.agent,
            zone=self,
            output_space_size=200,
            grid_size=20
        )

        self.distance = EncoderArea.add(
            name=AREA_NAME_DISTANCE,
            agent=self.agent,
            zone=self,
            min_inputs=1,
            surprise_level=0,
            recognition_threshold=0.9,
            accepts_dopamine_from=[AREA_NAME_DISTORTION]
        )

        self.container.add_connection(source=self.distance_receptive_area, target=self.distance)

        self.distance_change_receptive_area = SpatialReceptiveArea.add(
            name='distance change receptive',
            agent=self.agent,
            zone=self,
            output_space_size=100
        )

        self.distance_change = EncoderArea.add(
            name=AREA_NAME_DISTANCE_CHANGE,
            agent=self.agent,
            zone=self,
            min_inputs=1,
            surprise_level=0,
            accepts_dopamine_from=[AREA_NAME_DISTORTION]
        )
        self.container.add_connection(source=self.distance_change_receptive_area, target=self.distance_change)

    def activate_on_body(self, body_data, prev_body_data, data):
        body_shape_distorted = self._body_shape_distorted(data)
        # body_shape_distorted = body_data['overlay'] == True or (prev_body_data and prev_body_data['overlay'] == True)
        self.body_distortion.activate_on_body(body_shape_distorted)

        if body_data['name'] != 'hand':
            self._activate_velocity(body_data)
            self._activate_eye_shift_areas(body_data, prev_body_data)

        self.primitives_receptive_area.activate_on_body(body_data['general_presentation'], body_data['name'])

        self._activate_eye_shift_distance_area(body_data, prev_body_data)

        self._activate_eye_shift_distance_change_area(body_data, prev_body_data)

    def _activate_velocity(self, body_data):
        max_velocity = 5
        if 'offset' not in body_data:
            return

        x = body_data['offset'][0] / max_velocity
        y = body_data['offset'][1] / max_velocity
        x = min(1, x) if x > 0 else max(-1, x)
        y = min(1, y) if y > 0 else max(-1, y)

        velocity_left, velocity_right = (0, x) if x > 0 else (-x, 0)
        velocity_up, velocity_down = (0, y) if y > 0 else (-y, 0)

        self.velocity_left.activate_on_body(velocity_left)
        self.velocity_right.activate_on_body(velocity_right)
        self.velocity_up.activate_on_body(velocity_up)
        self.velocity_down.activate_on_body(velocity_down)

    def _body_shape_distorted(self, data):
        threshold = 20
        there_is_circle = False
        hand_data = [d for d in data if d['name'] == 'hand'][0]
        for body_data in data:
            if body_data['name'] == 'circle':
                there_is_circle = True
            if body_data['name'] != 'hand':
                shift_x = abs(hand_data['center'][0] - body_data['center'][0])
                shift_y = abs(hand_data['center'][1] - body_data['center'][1])
                if math.sqrt(shift_x ** 2 + shift_y ** 2) <= threshold:
                    return True
        if not there_is_circle:
            return True
        return False

    def _activate_eye_shift_areas(self, body_data, prev_body_data):
        from agent import ROOM_WIDTH, ROOM_HEIGHT

        half_width = ROOM_WIDTH // 2
        half_height = ROOM_HEIGHT // 2
        if prev_body_data:
            shift_x = body_data['center'][0] - prev_body_data['center'][0]
            shift_y = body_data['center'][1] - prev_body_data['center'][1]
        else:
            return

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

    def _activate_eye_shift_distance_area(self, body_data, prev_body_data):
        from agent import ROOM_WIDTH, ROOM_HEIGHT

        half_width = ROOM_WIDTH // 3
        half_height = ROOM_HEIGHT // 3

        if prev_body_data:
            shift_x = body_data['center'][0] - prev_body_data['center'][0]
            shift_y = body_data['center'][1] - prev_body_data['center'][1]
        else:
            return

        distance_norm = int(math.sqrt(half_height ** 2 + half_width ** 2))
        distance = int(math.sqrt(shift_x ** 2 + shift_y ** 2))
        distance = distance / distance_norm
        self.distance_receptive_area.activate_on_body(distance)

    def _activate_eye_shift_distance_change_area(self, body_data, prev_body_data):

        if prev_body_data:
            shift_x = body_data['center'][0] - prev_body_data['center'][0]
            shift_y = body_data['center'][1] - prev_body_data['center'][1]
        else:
            return

        body_name = body_data['name']
        if body_name == 'hand':
            body_name = prev_body_data['name']

        distance = int(math.sqrt(shift_x ** 2 + shift_y ** 2))
        if distance > 1000:
            return

        if body_name in self.distance_to_bodies:
            prev_distance = self.distance_to_bodies[body_name]
        else:
            prev_distance = 10000000

        if distance < prev_distance:
            value = 1
        elif distance == prev_distance:
            value = 0.5
        else:
            value = 0
        self.distance_to_bodies[body_name] = distance
        self.distance_change_receptive_area.activate_on_body(value)

    def on_area_updated(self, area: NeuralArea):
        if area == self.velocity and area.output:
            self.agent.on_message({
                'message': 'attention-strategy',
                'strategy': 'focus',
            })
