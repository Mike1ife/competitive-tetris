import numpy as np
from tetris import Piece
from collections import deque


class Pathfinder:
    """Model to output a series of commands to reach the destination"""

    # The difficulty in a well-rounded pathfinder is the discrepency
    # between "down" behavior
    # Our game runs continuously (FPS)
    # BFS runs discretely
    # So it's hard to decode BFS sequence into the sequence we want
    # the agent to execute in real-time game

    def __init__(self):
        pass

    def get_actions(self, board: np.ndarray, piece: Piece) -> list:
        actions = []
        rotations = self._get_rotations(piece.shape)
        current_rotation_id = self._get_rotation_id(piece.shape, rotations)

        _, cols = board.shape

        # Simply move horizontally and rotate some times
        # then hard drop
        for rotation_id, shape in enumerate(rotations):
            for col in range(cols):
                dropped_row = self._drop(board, shape, piece.row, col)
                if not dropped_row:
                    continue

                board_result = self._place(board, shape, dropped_row, col)
                board_result = self._clear_lines(board)

                sequence = self._build_sequence(
                    piece, current_rotation_id, rotation_id, col
                )
                actions.append(
                    {
                        "row": dropped_row,
                        "col": col,
                        "shape": shape,
                        "sequence": sequence,
                        "board_result": board_result,
                    }
                )

        return actions

    def _get_rotations(self, shape: np.ndarray) -> list:
        rotations = []
        current = shape.copy()
        for _ in range(4):
            rotations.append(current)
            current = np.rot90(current, axes=(1, 0))
        return rotations

    def _get_rotation_id(self, shape: np.ndarray, rotations: list) -> int:
        for i, rotation in enumerate(rotations):
            if np.array_equal(shape, rotation):
                return i
        return 0

    def _can_move_to(
        self, board: np.ndarray, shape: np.ndarray, row: int, col: int
    ) -> bool:
        rows, cols = board.shape
        for r, c in np.argwhere(shape):
            new_row = row + r
            new_col = col + c
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                return False
            if board[new_row][new_col]:
                return False
        return True

    def _place(self, board: np.ndarray, shape: np.ndarray, row: int, col: int):
        new_board = board.copy()
        for r, c in np.argwhere(shape):
            new_board[row + r][col + c] = 1
        return new_board

    def _drop(
        self, board: np.ndarray, shape: np.ndarray, start_row: int, col: int
    ) -> int | None:
        row = start_row

        if not self._can_move_to(board, shape, row, col):
            return None

        while self._can_move_to(board, shape, row + 1, col):
            row += 1

        return row

    def _clear_lines(self, board: np.ndarray):
        rows, cols = board.shape
        remain = [row for row in range(rows) if not np.all(board[row])]
        clear_count = rows - len(remain)
        if clear_count == 0:
            return board
        empty = np.zeros((clear_count, cols), dtype=int)
        return np.vstack([empty, board[remain]])

    def _build_sequence(
        self,
        piece: Piece,
        current_rotation_id: int,
        target_rotation_id: int,
        target_col: int,
    ):
        sequence = []

        rotate_count = (target_rotation_id - current_rotation_id) % 4
        sequence.extend(["rotate"] * rotate_count)
        # negative: move left
        # positive: move right
        col_delta = target_col - piece.col
        if col_delta < 0:
            sequence.extend(["left"] * abs(col_delta))
        else:
            sequence.extend(["right"] * col_delta)

        # Hard Drop in the end
        sequence.append("drop")
        return sequence
