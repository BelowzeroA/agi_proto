from typing import List

from neuro.areas.confluence_area import ConfluenceArea
from neuro.areas.encoder_area import EncoderArea
from neuro.areas.reflex_area import ReflexArea
from neuro.areas.tactile_receptive_area import TactileReceptiveArea
from neuro.neural_zone import NeuralZone
from neuro.zones.reflex_zone import ReflexZone
from neuro.zones.tactile_zone import TactileZone
from neuro.zones.visual_recognition_zone import VisualRecognitionZone

AREA_NAME_TOUCH = 'touch'


class ConfluenceZone(NeuralZone):
    """
    Manages the areas of abstract representations
    """
    def __init__(
            self,
            name: str,
            agent,
            visual: VisualRecognitionZone,
            tactile: TactileZone,
            reflex: ReflexZone):
        super().__init__(name, agent)
        receptive_areas = [area for area in visual.areas if isinstance(area, EncoderArea)]
        reflex_areas = [area for area in reflex.areas if isinstance(area, ReflexArea)]
        self.receptive_areas = receptive_areas
        self.reflex_areas = reflex_areas
        self._build_areas()

    def _create_confluence_area(
            self,
            receptive_area: EncoderArea,
            reflex_area: ReflexArea
    ):
        name = f'confluence {receptive_area} + {reflex_area}'
        area = ConfluenceArea.add(
            name=name,
            agent=self.agent,
            zone=self
        )
        self.container.add_connection(source=receptive_area, target=area)
        self.container.add_connection(source=reflex_area, target=area, source_output_property='active_reflex')

    def _build_areas(self):
        for receptive_area in self.receptive_areas:
            for reflex_area in self.reflex_areas:
                self._create_confluence_area(receptive_area, reflex_area)
