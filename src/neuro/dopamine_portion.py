

class DopaminePortion:
    """
    Portion of dopamine goes to different areas and provides a teaching signal
    """
    def __init__(self, value: int, source: 'NeuralArea'):
        self.value = value
        self.source = source