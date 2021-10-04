from typing import List, Dict

from neuro.dopamine_portion import DopaminePortion
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_pattern import NeuralPattern
from neuro.patterns_connection import PatternsConnection

MIN_ENERGY = 0.9
TRACE_INTERVAL = 6 * HyperParameters.network_steps_per_env_step
ENERGY_CHARGE_STEP = 0.03
NUM_ELEMENTS_IN_CHAIN = 3


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


class PatternBurstChain:

    def __init__(self, patterns: List[NeuralPattern], tick: int):
        self.hash = self._get_patterns_hash(patterns)
        self.last_activation_tick = tick
        self.hit_count = 1

    @staticmethod
    def _get_patterns_hash(patterns: List[NeuralPattern]):
        _hashed = [str(pattern._id) for pattern in patterns]
        return '|'.join(_hashed)


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
        self.chains = []

    def _check_if_chained(self, pattern) -> bool:
        current_tick = self.agent.network.current_tick
        chain_list = [bh[1] for bh in self.burst_history[-NUM_ELEMENTS_IN_CHAIN + 1:]]
        if len(chain_list) < NUM_ELEMENTS_IN_CHAIN - 1:
            return False
        chain_list.append(pattern)
        chain = PatternBurstChain(chain_list, current_tick)
        existing_chains = [c for c in self.chains if c.hash == chain.hash]
        if len(existing_chains) == 0:
            self.chains.append(chain)
            return False
        existing_chain = existing_chains[0]
        existing_chain.hit_count += 1
        chained = False
        if existing_chain.hit_count > 5 and existing_chain.last_activation_tick > current_tick - 200:
            chained = True
        existing_chain.last_activation_tick = current_tick
        return chained

    def process_pattern(self, pattern: NeuralPattern):
        current_tick = self.agent.network.current_tick
        trace_interval = 8 * HyperParameters.network_steps_per_env_step
        energies = [pe for pe in self.pattern_energies
                    if pe.pattern == pattern and pe.tick < current_tick - trace_interval and pe.value > 0]
        dope_value = 0
        for pe in energies:
            if pe.energy >= MIN_ENERGY:
                dope_value += pe.value
                pe.last_activation_tick = current_tick
                pe.control_tick = current_tick + TRACE_INTERVAL
                pe.energy = 0.0
                if self._check_if_chained(pe.pattern):
                    pe.energy = -3
                self.burst_history.append((current_tick, pe.pattern))

        return dope_value

    def update(self):
        current_tick = self.agent.network.current_tick
        input_patterns = [pattern for pattern in self.inputs if pattern]
        if len(input_patterns) == 0:
            return

        self.history[current_tick] = input_patterns

        dope_value = 0
        for pattern in input_patterns:
            dope_value += self.process_pattern(pattern)

        average_dope_value = dope_value // len(input_patterns)
        if average_dope_value > 0:
            self.zone.receive_self_induced_dope(average_dope_value)

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

    def receive_dope(self, dopamine_portions: List[DopaminePortion], self_induced=False):
        current_tick = self.agent.network.current_tick
        # untrace the traced patterns
        for pe in self.pattern_energies:
            for portion in dopamine_portions:
                if pe.control_tick and pe.control_tick < current_tick + TRACE_INTERVAL and \
                        pe.pattern.accepts_dopamine(portion) and pe.value <= portion.value:
                    pe.control_tick = 0

        for portion in dopamine_portions:
            dope_value = portion.value
            if dope_value < 2:
                continue
        
            processed_input_patterns = []
            start_tick = current_tick - 4 if self_induced else current_tick - 8
            for causing_tick in range(start_tick, current_tick - 2):
                if causing_tick in self.history:
                    patterns = self.history[causing_tick]
                    for pattern in patterns:
                        if pattern not in processed_input_patterns and pattern.accepts_dopamine(portion):
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

    def receive_inputs(self, input_patterns: List[NeuralPattern]):
        self.inputs = input_patterns





