from neuro.areas.encoder_area import EncoderArea
from neuro.areas.spatial_receptive_area import SpatialReceptiveArea
from neuro.areas.working_memory_cell import WorkingMemoryCell
from neuro.hyper_params import HyperParameters
from neuro.neural_area import NeuralArea
from neuro.neural_zone import NeuralZone


class VisualAttentionZone(NeuralZone):
    """
    Manages the attention system
    """
    def __init__(self, name: str, agent):
        super().__init__(name, agent)
        self.attention_locations = {}
        self._build_areas()

    def _build_areas(self):
        self.attention_location_horizontal = SpatialReceptiveArea.add(
            name='attention-horizontal',
            agent=self.agent,
            zone=self,
            grid_size=20,
        )

        self.attention_location_vertical = SpatialReceptiveArea.add(
            name='attention-vertical',
            agent=self.agent,
            zone=self,
            grid_size=20,
        )

        self.attention_location = EncoderArea.add(
            name='attention location',
            agent=self.agent,
            zone=self,
            output_space_size=HyperParameters.encoder_space_size,
            output_norm=HyperParameters.encoder_norm,
            min_inputs=2,
            surprise_level=0,
            recognition_threshold=0.99
        )
        self.attention_location.convey_new_pattern = True

        self.container.add_connection(source=self.attention_location_horizontal, target=self.attention_location)
        self.container.add_connection(source=self.attention_location_vertical, target=self.attention_location)

        self._add_working_memory()

    def _add_working_memory(self):
        for i in range(3):
            area = WorkingMemoryCell.add(
                name=f'working memory {i}',
                agent=self.agent,
                zone=self,
            )

            self.container.add_connection(
                source=self.attention_location,
                target=area
            )

    def _activate_attention_location(self, body_data):
        from agent import ROOM_WIDTH, ROOM_HEIGHT

        x = body_data['center'][0] / ROOM_WIDTH
        y = body_data['center'][1] / ROOM_HEIGHT

        if 0 <= x <= 1.0 and 0 <= y <= 1.0:
            self.attention_location_horizontal.activate_on_body(x)
            self.attention_location_vertical.activate_on_body(y)

    def activate_on_body(self, body_data):
        self._activate_attention_location(body_data)
        # self._activate_velocity(body_data)

    def on_area_updated(self, area: NeuralArea):
        # if area == self.velocity and area.output:
        #     self.agent.on_message({
        #         'message': 'attention-strategy',
        #         'strategy': 'focus',
        #     })
        if area == self.attention_location and area.output:
            self.agent.on_message({
                'message': 'attention-location',
                'location': area.output.data,
            })
            self.attention_locations[self.agent.network.current_tick] = area.output.data
