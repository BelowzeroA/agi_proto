from neuro.areas.encoder_area import EncoderArea
from neuro.areas.tactile_receptive_area import TactileReceptiveArea
from neuro.neural_zone import NeuralZone

AREA_NAME_TOUCH = 'touch'


class TactileZone(NeuralZone):
    """
    Manages the areas related to tactile perception
    """
    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self._build_areas()

    @property
    def output_area(self):
        return self._touch_area

    def _build_areas(self):
        self._receptive_area = TactileReceptiveArea.add(
            name='tactile perception',
            agent=self.agent,
            zone=self,
        )

        self._touch_area = EncoderArea.add(
            name=AREA_NAME_TOUCH,
            agent=self.agent,
            zone=self,
            surprise_level=2,
            recognition_threshold=0.9,
            accepts_dopamine_from=['self']
        )

        self.container.add_connection(source=self._receptive_area, target=self._touch_area)

    def activate(self, data):
        self._receptive_area.activate(data)


