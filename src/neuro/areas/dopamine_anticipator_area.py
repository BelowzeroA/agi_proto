from typing import List, Dict

from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection

MIN_ENERGY = 0.9
TRACE_INTERVAL = 6 * HyperParameters.network_steps_per_env_step
ENERGY_CHARGE_STEP = 0.02


class PatternDopeEnergy:

    def __init__(self, pattern, value: int, tick: int):
        self.pattern = pattern
        self.value = value
        self.tick = tick
        self.last_activation_tick = 0
        self.control_tick = 0
        self.energy = 0.0

    def _repr(self):
        return f'({self.pattern}) value={self.value}, last_activation={self.last_activation_tick}'

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return self._repr()

    def set_value(self, value):
        self.value = max(0, value)


class DopamineAnticipatorArea(NeuralArea):

    def __init__(
            self,
            name: str,
            agent,
            zone,
    ):
        super().__init__(name=name, agent=agent, zone=zone)
        self.history = {}
        self.pattern_energies: List[PatternDopeEnergy] = []
        self.burst_history = []

    def update(self):
        current_tick = self.agent.network.current_tick
        input_pattern = self.inputs[0]
        if input_pattern is None:
            return

        self.history[current_tick] = input_pattern

        trace_interval = 8 * HyperParameters.network_steps_per_env_step
        energies = [pe for pe in self.pattern_energies
                    if pe.pattern == input_pattern and pe.tick < current_tick - trace_interval and pe.value > 0]
        dope_value = 0
        for pe in energies:
            if pe.energy >= MIN_ENERGY:
                dope_value += pe.value
                pe.last_activation_tick = current_tick
                pe.control_tick = current_tick + TRACE_INTERVAL
                pe.energy = 0.0
                self.burst_history.append((current_tick, pe.pattern))

        if dope_value > 0:
            self.zone.receive_self_induced_dope(dope_value)

        self._distress_patterns()
        self._refresh_patterns_energy()

    def _refresh_patterns_energy(self):
        for pe in self.pattern_energies:
            if pe.energy < MIN_ENERGY:
                pe.energy += ENERGY_CHARGE_STEP

    def _distress_patterns(self):
        current_tick = self.agent.network.current_tick
        for pe in self.pattern_energies:
            if pe.control_tick and pe.control_tick <= current_tick:
                pe.set_value(pe.value - 1)
                pe.control_tick = 0

    def receive_dope(self, dope_value: int, self_induced=False):
        current_tick = self.agent.network.current_tick
        # untrace the traced patterns
        for pe in self.pattern_energies:
            if pe.control_tick and pe.control_tick < current_tick + TRACE_INTERVAL and pe.value <= dope_value:
                pe.control_tick = 0

        if dope_value < 2:
            return
        
        processed_input_patterns = []
        start_tick = current_tick - 4 if self_induced else current_tick - 8
        for causing_tick in range(start_tick, current_tick - 2):
            if causing_tick in self.history:
                pattern = self.history[causing_tick]
                if pattern not in processed_input_patterns:
                    energy_recs = [pe for pe in self.pattern_energies if pe.pattern == pattern]
                    if energy_recs:
                        energy_rec = energy_recs[0]
                        energy_rec.set_value(dope_value)
                        if not self_induced:
                            energy_rec.energy = 1.0
                    else:
                        energy_rec = PatternDopeEnergy(
                            pattern=pattern,
                            value=dope_value,
                            tick=causing_tick,
                        )
                        self.pattern_energies.append(energy_rec)
                    processed_input_patterns.append(pattern)





