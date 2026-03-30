import numpy as np
from tensorflow import keras
from config import ROWS, COLS, TETROMINOS
from tetris import Piece

NUM_PIECES = len(TETROMINOS)


class Decider:
    """Chooses the best action using the trained DQN model"""

    def __init__(self, source: str):
        self.model = keras.models.load_model(f"./models/{source}")

    def get_sequence(self, actions: list, piece: Piece, opp_agg: int = 0,
                     hold_piece=None, hold_used=False) -> list:
        if not actions:
            return []

        states = np.array([self._make_state(a, piece, opp_agg, hold_piece, hold_used) for a in actions])
        qs = self.model.predict(states, verbose=0).flatten()
        return actions[int(np.argmax(qs))]["sequence"]

    def _make_state(self, action: dict, piece: Piece, opp_agg: int,
                    hold_piece=None, hold_used=False) -> np.ndarray:
        board = action["board_result"]
        lines_cleared = action["lines_cleared"]

        heights = np.zeros(COLS, dtype=int)
        for col in range(COLS):
            for row in range(ROWS):
                if board[row][col]:
                    heights[col] = ROWS - row
                    break
        agg = int(heights.max())
        bump = int(sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1)))
        holes = sum(
            1
            for c in range(COLS)
            if heights[c]
            for r in range(ROWS - heights[c], ROWS)
            if not board[r][c]
        )
        own = np.array([agg, holes, bump, lines_cleared], dtype=np.float32)

        piece_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        piece_oh[piece.color_id - 1] = 1.0

        hold_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        if hold_piece is not None:
            _, hold_color_id = hold_piece
            hold_oh[hold_color_id - 1] = 1.0

        hold_avail = np.array([0.0 if hold_used else 1.0], dtype=np.float32)

        return np.concatenate([own, piece_oh, hold_oh, [opp_agg], hold_avail])