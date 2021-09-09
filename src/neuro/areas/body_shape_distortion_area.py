import random

from neuro.areas.receptive_area import ReceptiveArea
from neuro.hyper_params import HyperParameters


class BodyShapeDistortionArea(ReceptiveArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
    ):
        super().__init__(name, agent, zone)
        self.current_state = False
        self.counter = 0
        self.last_reset_tick = 0

    def activate_on_body(self, data):
        self.current_state = data

    def update(self):
        current_tick = self.agent.network.current_tick
        if self.current_state:
            self.counter += 1
        else:
            if self.counter > 0:
                self.last_reset_tick = current_tick
            self.counter = 0

        if current_tick > 20000:
            return

        if self.counter == 1 and current_tick - self.last_reset_tick > 10 * HyperParameters.network_steps_per_env_step:
            self.agent.on_message({
                'message': 'pattern_created',
                'surprise_level': 3,
                'area': self,
            })
