from neuro.areas.representation_area import RepresentationArea
from neuro.areas.receptive_area import ReceptiveArea
from neuro.container import NeuroContainer
from neuro.network import Network
from neuro.neural_area import NeuralArea


def main():
    container = NeuroContainer()
    receptive_area = ReceptiveArea(name='receptive', container=container)
    presentation_area = RepresentationArea(name='representations', container=container)
    container.add_area(receptive_area)
    container.add_area(presentation_area)
    network = Network(container=container)
    network.run()


if __name__ == '__main__':
    main()
