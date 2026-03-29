import pygame
import random
import numpy as np
from config import (
    ROWS,
    COLS,
    TETROMINOS,
    TETROMINOS_NAMES,
    WALL_KICKS,
    COLORS,
    FALL_INTERVAL,
    ACCELERATE_INTERVAL,
    CELL_SIZE,
    BOARD_H,
    BOARD_W,
    MAX_GARBAGE_HOLE,
    SCORE_TABLE,
)


class Piece:
    def __init__(self, shape: np.ndarray, color_id: int):
        self.shape = shape.copy()
        self.color_id = color_id
        # Generate at the middle of the top
        self.row = 0
        self.col = COLS // 2 - shape.shape[1] // 2
        self.rotation_id = 0

    def copy(self):
        p = Piece(self.shape, self.color_id)
        p.row = self.row
        p.col = self.col
        p.rotation_id = self.rotation_id
        return p


class Tetris:
    def __init__(self, x_offset: int, commands: dict):
        # render window offset (for placement)
        self.x_offset = x_offset
        self.commands = commands
        # only track placed pieces / garbage
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.cell_colors = np.zeros((ROWS, COLS), dtype=int)
        self.bag = list()  # 7-bag system
        self.piece: Piece = self._respawn_piece()
        self.opponent: Tetris = None  # Also a Tetris object
        self.game_over = False
        self.score = 0
        self._fall_timer = 0
        self.total_lines_cleared = 0
        self.accelerate = False

    def respawn_garbage_lines(self, count: int):
        """Respawn garbage lines"""
        garbage_lines = []
        for _ in range(count):
            row = np.ones(COLS, dtype=int)
            hole_count = random.randint(1, MAX_GARBAGE_HOLE)
            hole_cols = random.sample(range(COLS), hole_count)
            row[hole_cols] = 0
            garbage_lines.append(row)

        garbage_lines = np.array(garbage_lines)
        self.board = np.vstack((self.board[count:], garbage_lines))
        self.cell_colors = np.vstack((self.cell_colors[count:], garbage_lines * 8))

    def handle_event(self, event: pygame.event.Event):
        """This method is used to handle user's key event"""
        if self.game_over:
            return

        down_key = self.commands["down"][0]

        if event.type == pygame.KEYDOWN and event.key == down_key:
            self.accelerate = True
        elif event.type == pygame.KEYUP and event.key == down_key:
            self.accelerate = False
        elif event.type == pygame.KEYDOWN:
            for command, (key, _) in self.commands.items():
                if event.key == key:
                    self.execute(command)
                    break

    def execute(self, command: str):
        """This method is used to execute agent's command input"""
        if self.game_over or command not in self.commands:
            return

        # execute pygame event
        event, unicode = self.commands[command]
        new_event = pygame.event.Event(event, unicode=unicode, key=ord(unicode))
        pygame.event.post(new_event)

        # update piece info
        piece = self.piece
        if command == "left" and self._can_move_to(
            piece.shape, piece.row, piece.col - 1
        ):
            piece.col -= 1
        elif command == "right" and self._can_move_to(
            piece.shape, piece.row, piece.col + 1
        ):
            piece.col += 1
        elif command == "rotate":
            rotated_shape = self._try_rotate(piece, piece.row, piece.col)
            if rotated_shape is not None:
                new_shape, new_row, new_col = rotated_shape
                piece.shape = new_shape
                piece.row = new_row
                piece.col = new_col
                piece.rotation_id = (piece.rotation_id + 1) % 4
        elif command == "drop":
            while self._can_move_to(piece.shape, piece.row + 1, piece.col):
                piece.row += 1
            self._place()

    def update(self, delta: int):
        """Updating board with auto drop"""
        if self.game_over:
            return

        # automatically drop the piece with timer
        interval = ACCELERATE_INTERVAL if self.accelerate else FALL_INTERVAL
        self._fall_timer += delta
        if self._fall_timer >= interval:
            self._fall_timer = 0
            piece = self.piece
            # move downward, if cannot, place it
            if self._can_move_to(piece.shape, piece.row + 1, piece.col):
                piece.row += 1
            else:
                self._place()

    def render(self, screen: pygame.surface.Surface, font: pygame.font.Font):
        """Render the board"""
        # draw board (rect) on screen (window)
        for row in range(ROWS):
            for col in range(COLS):
                color_id = self.cell_colors[row, col]
                color = COLORS[color_id]
                rect = pygame.Rect(
                    self.x_offset + col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE - 1,
                    CELL_SIZE - 1,
                )
                pygame.draw.rect(screen, color, rect)

        # draw current piece
        piece = self.piece
        for r, c in np.argwhere(piece.shape):
            rect = pygame.Rect(
                self.x_offset + (piece.col + c) * CELL_SIZE,
                (piece.row + r) * CELL_SIZE,
                CELL_SIZE - 1,
                CELL_SIZE - 1,
            )
            pygame.draw.rect(screen, COLORS[piece.color_id], rect)

        # display score
        label = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(label, (self.x_offset + 4, 4))

        # display game over
        if self.game_over:
            modal = pygame.Surface(
                (BOARD_W, BOARD_H), pygame.SRCALPHA
            )  # SRCALPHA: transparency support
            modal.fill((0, 0, 0, 160))
            screen.blit(modal, (self.x_offset, 0))

            message = font.render("GAME OVER", True, (255, 60, 60))
            screen.blit(
                message,
                (self.x_offset + BOARD_W // 2 - message.get_width() // 2, BOARD_H // 2),
            )

    def _respawn_piece(self) -> Piece:
        """Respawn a new piece once we start the game or place a piece"""
        # 7-bag system means we put all unique pieces into the bag and shuffle it
        # each time we take one piece from the bag until it's empty,
        # then we refill and shuffle again
        if not self.bag:
            self.bag = TETROMINOS.copy()
            random.shuffle(self.bag)

        shape, color_id = self.bag.pop()
        return Piece(shape, color_id)

    def _can_move_to(self, shape: np.ndarray, row: int, col: int) -> bool:
        """Check whether we can move shape to (row, col)"""
        for r, c in np.argwhere(shape):
            # r, c represent local coordinate of the shape where shape[r, c] = 1
            # shape position + local coordinate = position of TETROMINOS
            new_row = row + r
            new_col = col + c
            # return False if collide
            if new_row < 0 or new_row >= ROWS or new_col < 0 or new_col >= COLS:
                return False
            if self.board[new_row][new_col]:
                return False
        return True

    def _get_kick_offsets(
        self, piece_id: int, current_rotation_id: int, target_rotation_id: int
    ) -> list:
        return WALL_KICKS[TETROMINOS_NAMES[piece_id]][
            (current_rotation_id, target_rotation_id)
        ]

    def _try_rotate(
        self, piece: Piece, row: int, col: int
    ) -> tuple[np.ndarray, int, int] | None:
        shape = piece.shape
        rotated_shape = np.rot90(shape, axes=(1, 0))
        for drow, dcol in self._get_kick_offsets(
            piece.color_id, piece.rotation_id, (piece.rotation_id + 1) % 4
        ):
            new_row = row + drow
            new_col = col + dcol
            if self._can_move_to(rotated_shape, new_row, new_col):
                return rotated_shape, new_row, new_col
        return None

    def _place(self):
        """Place a piece"""
        piece = self.piece
        for r, c in np.argwhere(piece.shape):
            row = piece.row + r
            col = piece.col + c

            if row < 0:
                self.game_over = True
                return

            self.board[row][col] = 1
            self.cell_colors[row][col] = piece.color_id

        cleared_count, new_board, new_cell_colors = self._clear_lines()
        self.board = new_board
        self.cell_colors = new_cell_colors
        if cleared_count and self.opponent:
            self.opponent.respawn_garbage_lines(cleared_count)

        self.piece = self._respawn_piece()
        # game is automatically over if respawned piece cannot be placed
        if not self._can_move_to(self.piece.shape, self.piece.row, self.piece.col):
            self.game_over = True

    def _clear_lines(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Clear lines (return number of cleared lines, excluding garbage lines, and resultant board)"""
        # separately count complete normal lines and garbage lines
        complete_lines = []
        garbage_line_count = 0
        for row in range(ROWS):
            if all(self.board[row]):
                complete_lines.append(row)
                # use color to check if it's a garbage line
                if 8 in self.cell_colors[row]:
                    garbage_line_count += 1

        self.total_lines_cleared += len(complete_lines)
        self.total_lines_cleared += len(complete_lines)
        line_cleared = len(complete_lines) - garbage_line_count
        self.score += SCORE_TABLE.get(line_cleared, 800)

        # Re-fill the board by placing empty lines on top of remaining rows
        remaining_lines = [row for row in range(ROWS) if row not in complete_lines]
        empty_lines = np.zeros((len(complete_lines), COLS), dtype=int)

        # Stack vertically (empty -> remaining)
        new_board = np.vstack((empty_lines.copy(), self.board[remaining_lines]))
        new_cell_colors = np.vstack(
            (empty_lines.copy(), self.cell_colors[remaining_lines])
        )

        return line_cleared, new_board, new_cell_colors

    # NOTE: I think we need to include the number of lines the opponent current have into our reward
    # The intuition is if we can end the opponent with 1 or 2 extra garbage lines, it should
    # be rewarded at maximum.
    # Also, the training agent should simulate the action itself. It's the not responsibility of Tetris.

    def _get_heights(self) -> np.ndarray:
        """Get the height of the board"""
        heights = np.zeros(COLS, dtype=int)
        for col in range(COLS):
            for row in range(ROWS):
                if self.board[row][col]:
                    heights[col] = ROWS - row
                    break
        return heights

    def get_game_state(self) -> dict:
        """Return game state properties needed for reward calculation"""
        heights = self._get_heights()
        return {
            "heights": heights,
            "aggregate_height": int(heights.max()),
            "bumpiness": self._get_bumpiness(heights),
            "holes": self._get_holes(heights),
        }

    def _get_bumpiness(self, heights: np.ndarray = None) -> int:
        """Get bumpiness of the board"""
        if heights is None:
            heights = self._get_heights()
        bumpiness = 0
        for col in range(COLS - 1):
            bumpiness += abs(heights[col] - heights[col + 1])
        return bumpiness

    def _get_holes(self, heights: np.ndarray = None) -> int:
        """Get holes in the board"""
        if heights is None:
            heights = self._get_heights()
        holes = 0
        for col in range(COLS):
            if heights[col] != 0:
                for row in range(ROWS - heights[col], ROWS):
                    if not self.board[row][col]:
                        holes += 1
        return holes

    # TODO: ADD MORE IF NEEDED
    # not for now