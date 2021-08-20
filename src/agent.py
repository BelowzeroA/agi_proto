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
        self.actions = {a: 0 for a in ACTIONS}
        self.attended_location_pattern = None
        self.attention_strategy = 'loop'
        self.attention_spot = None
        self.last_attention_loop_switch_tick = 0

    def _build_network(self):
        self.visual_recognition = VisualRecognitionZone.add(name='VR', agent=self)
        self.visual_attention = VisualAttentionZone.add(name='VA', agent=self)
        self.motor = MotorZone.add(name='MO', agent=self)

    def on_message(self, data: dict):
        message = data['message']
        if message == 'pattern_created':
            self.surprise += 1
        elif message == 'hand_move':
            self.actions[data['action_id']] = data['action_value']
        elif message == 'attention-strategy':
            if data['strategy'] == 'focus':
                self.attention_strategy = 'focus'
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

        self._serial_activate_on_body(body_data, prev_body_data)

    def activate_receptive_areas(self, data):
        if self.attention_strategy == 'loop':
            self.loop_strategy(data)
        else:
            self.loop_strategy(data)

    def _get_previous_body_index(self, data):
        if len(data) < 2:
            return None
        if self.focused_body_idx == 0:
            return len(data) - 1
        else:
            return self.focused_body_idx - 1

    def _serial_activate_on_body(self, body_data, prev_body_data):
        self.visual_recognition.activate_on_body(body_data, prev_body_data)
        self.visual_attention.activate_on_body(body_data)

        self.surprise = 0

        self.network.verbose = False
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
        return {
            'surprise': self.surprise,
            'actions': self.actions,
            'attention-spot': {'x': attention_x, 'y': attention_y}
        }

