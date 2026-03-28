import numpy as np
from train import make_state
from tensorflow import keras

class Decider:
    """Model that chooses the most optimal legal position to place the piece"""

    def __init__(self, source):
        self.model = keras.models.load_model(source)

    def get_sequence(self, actions, piece, opp_agg) -> list:
        states = np.array([
            make_state(a["board_result"], a["lines_cleared"], piece, opp_agg)
            for a in actions
        ])

        qs = self.model.predict(states, verbose=0).flatten()

        return actions[int(np.argmax(qs))]["sequence"]