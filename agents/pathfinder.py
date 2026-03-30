import numpy as np
from tetris import Piece
from config import TETROMINOS_NAMES, WALL_KICKS


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

        # Simply move horizontally and rotate sometimes
        # then hard drop
        for rotation_id, shape in enumerate(rotations):
            for col in range(cols):
                dropped_row = self._drop(board, shape, piece.row, col)
                if dropped_row is None:
                    continue

                board_placed = self._place(board, shape, dropped_row, col)
                lines_cleared, board_result = self._clear_lines(board_placed)

                sequence = self._build_sequence(
                    board, piece, current_rotation_id, rotation_id, col
                )
                if sequence is None:
                    continue

                actions.append(
                    {
                        "row": dropped_row,
                        "col": col,
                        "shape": shape,
                        "sequence": sequence,
                        "lines_cleared": lines_cleared,
                        "board_result": board_result,
                        "color_id": piece.color_id - 1,  # 0-indexed for one-hot
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
            if shape.shape == rotation.shape and np.array_equal(shape, rotation):
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

    def _clear_lines(self, board: np.ndarray) -> tuple[int, np.ndarray]:
        rows, cols = board.shape
        remain = [row for row in range(rows) if not np.all(board[row])]
        clear_count = rows - len(remain)
        if clear_count == 0:
            return 0, board
        empty = np.zeros((clear_count, cols), dtype=int)
        return clear_count, np.vstack([empty, board[remain]])

    def _try_rotate(
        self, board: np.ndarray, piece_color_id: int, shape: np.ndarray,
        rotation_id: int, row: int, col: int
    ) -> tuple[np.ndarray, int, int, int] | None:
        rotated_shape = np.rot90(shape, axes=(1, 0))
        target_rotation_id = (rotation_id + 1) % 4
        kick_offsets = WALL_KICKS[TETROMINOS_NAMES[piece_color_id]][
            (rotation_id, target_rotation_id)
        ]
        for drow, dcol in kick_offsets:
            new_row = row + drow
            new_col = col + dcol
            if self._can_move_to(board, rotated_shape, new_row, new_col):
                return rotated_shape, new_row, new_col, target_rotation_id
        return None

    def _build_sequence(
        self,
        board: np.ndarray,
        piece: Piece,
        current_rotation_id: int,
        target_rotation_id: int,
        target_col: int,
    ):
        sequence = []
        sim_shape = piece.shape.copy()
        sim_row = piece.row
        sim_col = piece.col
        sim_rot = current_rotation_id

        rotate_count = (target_rotation_id - current_rotation_id) % 4
        for _ in range(rotate_count):
            result = self._try_rotate(
                board, piece.color_id, sim_shape, sim_rot, sim_row, sim_col
            )
            if result is None:
                return None
            sim_shape, sim_row, sim_col, sim_rot = result
            sequence.append("rotate")

        col_delta = target_col - sim_col
        if col_delta < 0:
            for _ in range(abs(col_delta)):
                if self._can_move_to(board, sim_shape, sim_row, sim_col - 1):
                    sim_col -= 1
                    sequence.append("left")
                else:
                    return None
        else:
            for _ in range(col_delta):
                if self._can_move_to(board, sim_shape, sim_row, sim_col + 1):
                    sim_col += 1
                    sequence.append("right")
                else:
                    return None

        # Hard Drop in the end
        sequence.append("drop")
        return sequence