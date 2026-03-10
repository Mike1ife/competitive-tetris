import pygame
import random
import numpy as np
from config import (
    ROWS,
    COLS,
    TETROMINOS,
    COLORS,
    FALL_INTERVAL,
    CELL_SIZE,
    BOARD_H,
    BOARD_W,
    MAX_GARBAGE_HOLE,
)


class Piece:
    def __init__(self, shape: np.ndarray, color_id: int):
        self.shape = shape.copy()
        self.color_id = color_id
        # Generate at the middle of the top
        self.row = 0
        self.col = COLS // 2 - shape.shape[1] // 2

    def rotate(self):
        self.shape = np.rot90(self.shape).copy()


class Tetris:
    def __init__(self, x_offset: int, commands: dict):
        self.x_offset = x_offset
        self.commands = commands

        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.cell_colors = np.zeros((ROWS, COLS), dtype=int)
        self.piece = self._respawn_piece()
        self.opponent: Tetris = None  # Also Tetris object
        self.game_over = False
        self.score = 0
        self._fall_timer = 0

    # Respawn new piece once we start the game or place a piece
    def _respawn_piece(self) -> Piece:
        shape, color_id = random.choice(TETROMINOS)
        return Piece(shape, color_id)

    # Check whether we can move shape to (row, col)
    def _can_move_to(self, shape: np.ndarray, row: int, col: int) -> bool:
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

    # Place a piece
    def _place(self):
        piece = self.piece
        for r, c in np.argwhere(piece.shape):
            row = piece.row + r
            col = piece.col + c

            if row < 0:
                self.game_over = True
                return

            self.board[row][col] = 1
            self.cell_colors[row][col] = piece.color_id

        cleared_count = self._clear_lines()
        if cleared_count and self.opponent:
            self.opponent.respawn_garbage_lines(cleared_count)

        self.piece = self._respawn_piece()
        # game is automatically over if respawned piece cannot be placed
        if not self._can_move_to(self.piece.shape, self.piece.row, self.piece.col):
            self.game_over = True

    # Clear lines (return number of cleared lines, excluding garbage lines)
    def _clear_lines(self) -> int:
        # separately count complete normal lines and garbage lines
        complete_lines = []
        garbage_line_count = 0
        for row in range(ROWS):
            if all(self.board[row]):
                complete_lines.append(row)
                # use color to check if it's a garbage line
                if self.cell_colors[row][0] == 8:
                    garbage_line_count += 1

        if not complete_lines:
            return 0

        self.score += len(complete_lines) - garbage_line_count
        # Re-fill the board by placing empty lines on top of remaining rows
        remaining_lines = [row for row in range(ROWS) if row not in complete_lines]
        empty_lines = np.zeros((len(complete_lines), COLS), dtype=int)

        # Stack vertically (empty -> remaining)
        self.board = np.vstack((empty_lines.copy(), self.board[remaining_lines]))
        self.cell_colors = np.vstack(
            (empty_lines.copy(), self.cell_colors[remaining_lines])
        )

        return len(complete_lines) - garbage_line_count

    # Respawn garbage lines
    def respawn_garbage_lines(self, count: int):
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

    # Handle event
    # This method is used to handle user's key event
    def handle_event(self, event: pygame.event.Event):
        # KEYDOWN != K_DOWN
        if self.game_over or event.type != pygame.KEYDOWN:
            return

        for command, (key, _) in self.commands.items():
            if event.key == key:
                self.execute(command)
                break

    # Execute command
    # This method is used to execute agent's command input
    def execute(self, command: str):
        if command not in self.commands:
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
        elif command == "down" and self._can_move_to(
            piece.shape, piece.row + 1, piece.col
        ):
            piece.row += 1
        elif command == "rotate":
            rotated_shape = np.rot90(piece.shape.copy(), k=3)
            if self._can_move_to(rotated_shape, piece.row, piece.col):
                piece.shape = rotated_shape

    def update(self, delta: int):
        if self.game_over:
            return

        # automatically drop the piece with timer
        self._fall_timer += delta
        if self._fall_timer >= FALL_INTERVAL:
            self._fall_timer = 0
            piece = self.piece
            # move downward, if cannot, place it
            if self._can_move_to(piece.shape, piece.row + 1, piece.col):
                piece.row += 1
            else:
                self._place()

    def render(self, screen: pygame.surface.Surface, font: pygame.font.Font):
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
