import random


class NeuralArea:

    def __init__(self, name: str, container):
        self.name = name
        self.container = container
        self.report_activity = True

    def connect(self, area: 'NeuralArea', density: int):
        source_neurons = self.container.get_area_neurons(self)
        target_neurons = self.container.get_area_neurons(area)
        for source_neuron in source_neurons:
            target_neurons_selection = random.sample(target_neurons, density)
            for target_neuron in target_neurons_selection:
                connection = self.container.add_connection(source_neuron, target_neuron)

    def update(self):
        pass

    def firing_allowed(self, neuron: 'Neuron'):
        return True

    def firing_mandatory(self, neuron: 'Neuron'):
        return False

    def report(self):
        if self.report_activity:
            neurons = self.container.get_area_neurons(self)
            active_neurons = [n for n in neurons if n.firing]
            print(f'{self.name} ({len(active_neurons)}): {active_neurons}')


