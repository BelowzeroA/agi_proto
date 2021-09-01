import random

from neuro.areas.action_area import ActionArea
from neuro.areas.hand_motion_area import HandMotionArea
from neuro.areas.primitives_receptive_area import PrimitivesReceptiveArea
from neuro.areas.spatial_receptive_area import SpatialReceptiveArea
from neuro.container import Container
from neuro.areas.encoder_area import EncoderArea
from neuro.hyper_params import HyperParameters
from neuro.network import Network
from neuro.zones.motor_zone import MotorZone
from neuro.zones.reflex_zone import ReflexZone
from neuro.zones.visual_attention_zone import VisualAttentionZone
from neuro.zones.visual_recognition_zone import VisualRecognitionZone

ROOM_WIDTH = 640
ROOM_HEIGHT = 480
ACTIONS = ['move_left', 'move_right', 'move_up', 'move_down', 'mode']
ATTENTION_SPAN = 5


class Agent:

    def __init__(self):
        random.seed(0)
        self.container = Container()
        self._build_network()
        self.network = Network(container=self.container, agent=self)
        self.focused_body_idx = None
        self.surprise = 0
        self.surprise_history = {}
        self.actions = {a: 0 for a in ACTIONS}
        self.attended_location_pattern = None
        self.attention_strategy = 'loop'
        self.attention_spot = None
        self.last_attention_loop_switch_tick = 0
        self.last_motion_tick = 0
        self.moving_body_start_tick = 0
        self.moving_body = None
        self.last_attended_body = None
        self.body_cache = {}

    def _build_network(self):
        self.visual_recognition = VisualRecognitionZone.add(name='VR', agent=self)
        self.visual_attention = VisualAttentionZone.add(name='VA', agent=self)
        self.motor = MotorZone.add(name='MO', agent=self)
        self.reflex = ReflexZone.add(
            name='RE',
            agent=self,
            motor_zone=self.motor,
            vr_zone=self.visual_recognition
        )

    def on_message(self, data: dict):
        current_tick = self.network.current_tick
        message = data['message']
        if message == 'pattern_created':
            self.surprise += data['surprise_level']
            if current_tick not in self.surprise_history:
                self.surprise_history[current_tick] = []
            self.surprise_history[current_tick].append(data)
        elif message == 'hand_move':
            self.actions[data['action_id']] = data['action_value']
        elif message == 'attention-strategy':
            if data['strategy'] == 'focus':
                self.attention_strategy = 'focus'
                self.last_motion_tick = current_tick
        elif message == 'attention-location':
            self.attention_spot = data['location']
        else:
            raise AttributeError(f'Unrecognized message: {message}')

    def _switch_focus(self, data):
        if self.focused_body_idx is None:
            self.focused_body_idx = 0
        elif self.focused_body_idx >= len(data) - 1:
            self.focused_body_idx = 0
        else:
            self.focused_body_idx += 1

    def loop_strategy(self, data):
        if len(data) == 0 or len(data) > 3:
            return

        current_tick = self.network.current_tick
        if current_tick == 0:
            self._switch_focus(data)
        elif current_tick - self.last_attention_loop_switch_tick > ATTENTION_SPAN:
            self._switch_focus(data)
            self.last_attention_loop_switch_tick = current_tick

        if self.focused_body_idx >= len(data):
            self.focused_body_idx = len(data) - 1

        if self.container.network.verbose:
            print(f'body #{self.focused_body_idx + 1}')

        previous_focused_body_idx = self._get_previous_body_index(data)
        prev_body_data = None
        if previous_focused_body_idx:
            prev_body_data = data[previous_focused_body_idx]
        body_data = data[self.focused_body_idx]

        prev_body_data = None
        self._serial_activate_on_body(body_data, prev_body_data, data)

    def _get_distance(self, x, y, body_data):
        dx = abs(x - body_data['center'][0])
        dy = abs(y - body_data['center'][1])
        return dx + dy

    def find_body_from_attention_spot(self, data):
        attention_x = int(self.attention_spot['attention-horizontal'] * ROOM_WIDTH)
        attention_y = int(self.attention_spot['attention-vertical'] * ROOM_HEIGHT)
        distances = []
        for i in range(len(data)):
            distance = self._get_distance(attention_x, attention_y, data[i])
            distances.append((data[i], distance))
        distances.sort(key=lambda x: x[1])
        return distances[0][0]

    def find_nearest_neighbor(self, data, body):
        x, y = body['center'][0], body['center'][1]
        distances = []
        for i in range(len(data)):
            if data[i] == body:
                continue
            distance = self._get_distance(x, y, data[i])
            distances.append((data[i], distance))
        distances.sort(key=lambda x: x[1])
        shortest_distance = distances[0][1]
        if shortest_distance > 150:
            return None
        return distances[0][0]

    def get_moving_body(self, data):
        for body_data in data:
            if body_data['offset'][0] != 0 or body_data['offset'][1] and body_data['name'] == 'hand':
                return body_data
        return None

    def _cache_bodies(self, data):
        for body_data in data:
            body_name = body_data['name']
            self.body_cache[body_name] = body_data['general_presentation']

    def focus_strategy(self, data):
        if len(data) == 0 or len(data) > 3:
            return

        if not self.body_cache:
            self._cache_bodies(data)

        current_tick = self.network.current_tick

        if self.moving_body_start_tick == current_tick - HyperParameters.network_steps_per_env_step:
            attended_body = self.last_attended_body
        else:
            attended_body = self.find_body_from_attention_spot(data)

        moving_body = self.get_moving_body(data)
        if moving_body is None:
            self._serial_activate_on_body(attended_body, None, data)
            return

        prev_attended_body = attended_body
        if current_tick - self.moving_body_start_tick > 3 * HyperParameters.network_steps_per_env_step:
            if moving_body == attended_body:
                attempted_body = self.find_nearest_neighbor(data, attended_body)
                if attempted_body:
                    attended_body = attempted_body
            else:
                attended_body = moving_body
            self.moving_body_start_tick = current_tick

        if prev_attended_body == attended_body:
            prev_attended_body = None

        self.last_attended_body = attended_body
        body_name = attended_body['name']
        if body_name in self.body_cache:
            attended_body['general_presentation'] = self.body_cache[body_name]

        self._serial_activate_on_body(attended_body, prev_attended_body, data)

    def activate_receptive_areas(self, data):
        current_tick = self.network.current_tick
        if current_tick - self.last_motion_tick > 7:
            self.attention_strategy = 'loop'
        if self.attention_strategy == 'loop':
            self.loop_strategy(data)
        else:
            self.focus_strategy(data)

    def _get_previous_body_index(self, data):
        if len(data) < 2:
            return None
        if self.focused_body_idx == 0:
            return len(data) - 1
        else:
            return self.focused_body_idx - 1

    def _serial_activate_on_body(self, body_data, prev_body_data=None, data=None):
        self.visual_recognition.activate_on_body(body_data, prev_body_data, data)
        self.visual_attention.activate_on_body(body_data)

        self.network.verbose = False
        for i in range(HyperParameters.network_steps_per_env_step):
            self.surprise = 0
            self.network.step()

    def env_step(self, data):
        self.actions = {a: 0 for a in ACTIONS}
        self.activate_receptive_areas(data)
        if self.container.network.verbose:
            print(f'Surprise: {self.surprise}')
        attention_x, attention_y = -1, -1
        if self.attention_spot:
            attention_x = int(self.attention_spot['attention-horizontal'] * ROOM_WIDTH)
            attention_y = int(self.attention_spot['attention-vertical'] * ROOM_HEIGHT)
        if self.last_attended_body:
            attention_x = self.last_attended_body['center'][0]
            attention_y = self.last_attended_body['center'][1]
        return {
            'current_tick': self.network.current_tick + 1,
            'surprise': self.surprise,
            'actions': self.actions,
            'attention-spot': {'x': attention_x, 'y': attention_y}
        }

