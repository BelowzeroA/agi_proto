import numpy as np


class AgentHand:

    def __init__(self, vec=None):
        if vec:
            self.hand_contour = np.array(vec)
        else:
            self.hand_contour = np.array(
                [[1.2, 0.0],
                 [1.0, 1.4],
                 [0.6, 0.3],
                 [0.8, 1.4],
                 [0.6, 1.6],
                 [0.0, 1.0],
                 [0.9, 3.1],
                 [1.7, 3.1],
                 [2.9, 2.3],
                 [1.8, 2.0],
                 [2.1, 0.4],
                 [1.5, 1.4], ])
        self.left = self.hand_contour[5][0]
        self.right = self.hand_contour[11][0]
        self.top = self.hand_contour[9][1]
        self.bottom = self.hand_contour[0][1]
        self._center = np.array([(self.left + self.right) / 2,
                                 (self.top + self.bottom) / 2])

    def move_hand(self, vec=(0, 0)):
        vec = np.array(vec)
        self.hand_contour = self.hand_contour + vec
        self.left = self.hand_contour[5][0]
        self.right = self.hand_contour[11][0]
        self.top = self.hand_contour[9][1]
        self.bottom = self.hand_contour[0][1]
        self._center = self._center + vec
        return self

    def __add__(self, other):
        return self.move_hand(other)

    def __iadd__(self, other):
        return self.move_hand(other)

    def __sub__(self, other):
        other = (-other[0], -other[1])
        return self.move_hand(other)

    def __isub__(self, other):
        other = (-other[0], -other[1])
        return self.move_hand(other)
