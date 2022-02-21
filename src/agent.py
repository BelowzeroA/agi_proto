import os
import random

from common.logger import Logger
from common.timer import Timer
from neuro.container import Container
from neuro.dopamine_portion import DopaminePortion
from neuro.hyper_params import HyperParameters
from neuro.network import Network
from neuro.zones.confluence_zone import ConfluenceZone
from neuro.zones.motor_zone import MotorZone
from neuro.zones.reflex_zone import ReflexZone
from neuro.zones.tactile_zone import TactileZone
from neuro.zones.visual_attention_zone import VisualAttentionZone
from neuro.zones.visual_recognition_zone import VisualRecognitionZone
from utils import path_from_root

ROOM_WIDTH = 640
ROOM_HEIGHT = 480
MAX_ATTENTION_DISTANCE = 500
ACTIONS = ['move_left', 'move_right', 'move_up', 'move_down', 'grab']
MACRO_ACTIONS = ['move', 'grab']

ATTENTION_SPAN = 5

log_path = os.path.join(path_from_root('logs'), 'log.txt')


class Agent:

    MAX_BODIES_NUM = 5

    """
    AGI agent is trying to adapt to the environment by observing it's state
    and learning useful reflexes to maximize dopamine
    """
    def __init__(self):
        random.seed(0)
        self.container = Container()
        self._build_network()
        self.network = Network(container=self.container, agent=self)
        self.logger = Logger(self, log_path)
        self.focused_body_idx = None
        self.surprise = 0
        self.dopamine_flow = {}
        self.actions = {a: 0 for a in ACTIONS}
        self.attended_location_pattern = None
        self.attention_strategy = 'loop'
        self.attention_spot = None
        self.last_attention_loop_switch_tick = 0
        self.last_motion_tick = 0
        self.moving_body_start_tick = 0
        self.moving_body = None
        self.last_attended_body = None
        self.last_attended_was_hand = False
        self.last_report_tick = 0
        self.body_cache = {}
        self.execution_timer = Timer()

    def _build_network(self):
        self.visual_recognition = VisualRecognitionZone.add(name='VR', agent=self)
        self.visual_attention = VisualAttentionZone.add(name='VA', agent=self)
        self.tactile = TactileZone(name='TA', agent=self)
        self.motor = MotorZone.add(name='MO', agent=self)
        self.reflex = ReflexZone.add(
            name='RE',
            agent=self,
            motor_zone=self.motor,
            vr_zone=self.visual_recognition,
            ta_zone=self.tactile
        )
        self.confluence = ConfluenceZone.add(
            name='CO',
            agent=self,
            visual=self.visual_recognition,
            tactile=self.tactile,
            reflex=self.reflex,
        )

    def on_message(self, data: dict):
        current_tick = self.network.current_tick
        message = data['message']
        if message == 'pattern_created':

            self.surprise += data['surprise_level']
            if current_tick not in self.dopamine_flow:
                self.dopamine_flow[current_tick] = []
            self.dopamine_flow[current_tick].append(DopaminePortion(data['surprise_level'], data['area']))

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

    def loop_strategy(self, packet):
        body_data = packet['data']
        if len(body_data) == 0 or len(body_data) > 5:
            return

        current_tick = self.network.current_tick
        if current_tick == 0:
            self._switch_focus(body_data)
        elif current_tick - self.last_attention_loop_switch_tick > ATTENTION_SPAN:
            self._switch_focus(body_data)
            self.last_attention_loop_switch_tick = current_tick

        if self.focused_body_idx >= len(body_data):
            self.focused_body_idx = len(body_data) - 1

        if self.container.network.verbose:
            print(f'body #{self.focused_body_idx + 1}')

        previous_focused_body_idx = self._get_previous_body_index(body_data)
        prev_body_data = None
        if previous_focused_body_idx:
            prev_body_data = body_data[previous_focused_body_idx]
        curr_body_data = body_data[self.focused_body_idx]

        prev_body_data = None
        self._serial_activate_on_body(curr_body_data, prev_body_data, packet)

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
        if shortest_distance > MAX_ATTENTION_DISTANCE:
            return None
        return distances[0][0]

    @staticmethod
    def body_out_of_room(body_data):
        center = body_data['center']
        return center[0] < 0 or center[0] > ROOM_WIDTH or center[1] < 0 or center[1] > ROOM_HEIGHT

    def get_moving_body(self, data):
        for body_data in data:
            if body_data['offset'][0] != 0 or body_data['offset'][1] \
                    and body_data['name'] != 'hand' and not self.body_out_of_room(body_data):
                return body_data
        for body_data in data:
            if body_data['offset'][0] != 0 or body_data['offset'][1] and body_data['name'] == 'hand':
                return body_data
        return None

    def _cache_bodies(self, data):
        for body_data in data:
            body_name = body_data['name']
            self.body_cache[body_name] = body_data['general_presentation']

    def focus_strategy_legacy(self, packet):
        data = packet['data']
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
            self._serial_activate_on_body(attended_body, None, packet)
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

        self._serial_activate_on_body(attended_body, prev_attended_body, packet)

    def focus_strategy(self, packet):
        data = packet['data']
        if len(data) == 0 or len(data) > Agent.MAX_BODIES_NUM:
            return

        current_tick = self.network.current_tick

        if not self.body_cache:
            self._cache_bodies(data)

        hand = [b for b in data if b['name'] == 'hand'][0]
        moving_body = self.get_moving_body(data)
        if moving_body == hand:
            moving_body = None

        prev_attended_body = None
        if current_tick - self.moving_body_start_tick > 3 * HyperParameters.network_steps_per_env_step:
            # if self.last_attended_was_hand:
            if self.last_attended_body and self.last_attended_body['name'] == 'hand':
                if moving_body is None:
                    attended_body = self.find_nearest_neighbor(data, hand)
                else:
                    attended_body = moving_body
            else:
                attended_body = hand
            self.moving_body_start_tick = current_tick
            if attended_body:
                prev_attended_body = self.last_attended_body
                self.last_attended_body = attended_body
                self.last_attended_was_hand = not self.last_attended_was_hand
        else:
            attended_body = self.last_attended_body

        if attended_body is None:
            attended_body = self.last_attended_body

        body_name = attended_body['name']
        if body_name in self.body_cache:
            attended_body['general_presentation'] = self.body_cache[body_name]

        if prev_attended_body and prev_attended_body['name'] == 'hand':
            prev_attended_body = hand

        if attended_body and attended_body['name'] == 'hand':
            attended_body = hand

        self._serial_activate_on_body(attended_body, prev_attended_body, packet)

    def activate_receptive_areas(self, packet):
        current_tick = self.network.current_tick
        if current_tick - self.last_motion_tick > 7 * HyperParameters.network_steps_per_env_step:
            self.attention_strategy = 'loop'
        if current_tick < 30:
            self.attention_strategy = 'loop'
        if self.attention_strategy == 'loop':
            self.loop_strategy(packet)
        else:
            self.focus_strategy(packet)

    def _get_previous_body_index(self, data):
        if len(data) < 2:
            return None
        if self.focused_body_idx == 0:
            return len(data) - 1
        else:
            return self.focused_body_idx - 1

    def _serial_activate_on_body(self, body_data, prev_body_data=None, packet=None):
        self.visual_recognition.activate_on_body(body_data, prev_body_data, packet['data'])
        self.visual_attention.activate_on_body(body_data)
        self.tactile.activate(packet['mode'])

        self.network.verbose = False
        for i in range(HyperParameters.network_steps_per_env_step):
            self.surprise = 0
            self.network.step()

        self.network.reset_perception()

    def _convert_move_actions(self):
        if 'move' in self.actions:
            self.actions['move_left'] = self.actions['move']['left']
            self.actions['move_right'] = self.actions['move']['right']
            self.actions['move_up'] = self.actions['move']['up']
            self.actions['move_down'] = self.actions['move']['down']

    def _convert_attention_spot(self):
        attention_x, attention_y = -1, -1
        if self.attention_spot:
            attention_x = int(self.attention_spot['attention-horizontal'] * ROOM_WIDTH)
            attention_y = int(self.attention_spot['attention-vertical'] * ROOM_HEIGHT)
        if self.last_attended_body:
            attention_x = self.last_attended_body['center'][0]
            attention_y = self.last_attended_body['center'][1]

        return attention_x, attention_y

    def _log_body_positions(self, packet):
        circle_pos = None
        triangle_pos = None
        hand_pos = None
        for body_data in packet['data']:
            if body_data['name'] == 'circle':
                circle_pos = body_data['center']
            elif body_data['name'] == 'triangle':
                triangle_pos = body_data['center']
            elif body_data['name'] == 'hand':
                hand_pos = body_data['center']
        self.logger.write_content(f'Positions hand: {hand_pos}, circle: {circle_pos}, triangle: {triangle_pos}')

    def env_step(self, packet):
        current_tick = self.network.current_tick + 1

        if current_tick - self.last_report_tick > 98:
            self.execution_timer.show(f'current_tick x100: {(current_tick + 1) // 100}, last 100 ticks', flush=True)
            self.last_report_tick = current_tick
            self.execution_timer.start()

        self.actions = {a: 0 for a in ACTIONS}
        self.activate_receptive_areas(packet)

        if self.container.network.verbose:
            print(f'Surprise: {self.surprise}')

        self._log_body_positions(packet)

        self._convert_move_actions()

        self.logger.write()

        attention_x, attention_y = self._convert_attention_spot()

        return {
            'current_tick': self.network.current_tick + 1,
            'surprise': self.surprise,
            'actions': self.actions,
            'attention-spot': {'x': attention_x, 'y': attention_y}
        }
