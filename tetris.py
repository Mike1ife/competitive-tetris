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
    SCORE_TABLE,
    PREVIEW_W,
    DAS,
    ARR,
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
        self.hold_piece = None  # (shape, color_id) tuple
        self.hold_used = False  # can only hold once per piece
        self.opponent: Tetris = None  # Also a Tetris object
        self.game_over = False
        self.score = 0
        self.normal_lines_cleared = 0
        self.garbage_lines_cleared = 0
        self._fall_timer = 0
        self.accelerate = False

        # DAS/ARR state for left/right
        self._held_dir = None   # "left" or "right" or None
        self._das_timer = 0     # ms elapsed since key held
        self._das_charged = False  # True once DAS delay has passed
        self._arr_timer = 0     # ms elapsed since last ARR repeat

    def respawn_garbage_lines(self, count: int):
        """Respawn garbage lines with consistent hole column (Jstris style)"""
        if count == 0:
            return
        
        hole_col = random.randint(0, COLS - 1)
        garbage_lines = []
        for _ in range(count):
            row = np.ones(COLS, dtype=int)
            row[hole_col] = 0
            garbage_lines.append(row)

        garbage_lines = np.array(garbage_lines)
        self.board = np.vstack((self.board[count:], garbage_lines))
        self.cell_colors = np.vstack((self.cell_colors[count:], garbage_lines * 8))

    def hold(self):
        """Swap current piece with hold piece. Can only hold once per piece."""
        if self.hold_used:
            return
        self.hold_used = True
        if self.hold_piece is None:
            self.hold_piece = (self.piece.shape.copy(), self.piece.color_id)
            self.piece = self._respawn_piece()
        else:
            old_shape, old_color = self.hold_piece
            self.hold_piece = (self.piece.shape.copy(), self.piece.color_id)
            self.piece = Piece(old_shape, old_color)

    def _get_dir_for_key(self, key):
        """Return 'left' or 'right' if key matches, else None."""
        for cmd in ("left", "right"):
            if cmd in self.commands and self.commands[cmd][0] == key:
                return cmd
        return None

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
            # Check if this is a left/right key for DAS/ARR
            direction = self._get_dir_for_key(event.key)
            if direction:
                # Fire one immediate move
                self._move_piece(direction)
                # Start DAS tracking
                self._held_dir = direction
                self._das_timer = 0
                self._das_charged = False
                self._arr_timer = 0
            else:
                # Non-directional keys: rotate, hold, drop
                for command, (key, _) in self.commands.items():
                    if event.key == key:
                        self.execute(command)
                        break
        elif event.type == pygame.KEYUP:
            direction = self._get_dir_for_key(event.key)
            if direction and self._held_dir == direction:
                self._held_dir = None
                self._das_timer = 0
                self._das_charged = False
                self._arr_timer = 0

    def _move_piece(self, direction: str):
        """Move the current piece left or right by one cell."""
        piece = self.piece
        if direction == "left" and self._can_move_to(piece.shape, piece.row, piece.col - 1):
            piece.col -= 1
        elif direction == "right" and self._can_move_to(piece.shape, piece.row, piece.col + 1):
            piece.col += 1

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
        elif command == "hold":
            self.hold()
        elif command == "drop":
            while self._can_move_to(piece.shape, piece.row + 1, piece.col):
                piece.row += 1
            self._place()

    def update(self, delta: int):
        """Updating board with auto drop + DAS/ARR"""
        if self.game_over:
            return

        # DAS/ARR for held left/right
        if self._held_dir is not None:
            if not self._das_charged:
                self._das_timer += delta
                if self._das_timer >= DAS:
                    self._das_charged = True
                    self._arr_timer = 0
                    # Fire first auto-shift move (or snap if ARR=0)
                    if ARR == 0:
                        self._snap_piece(self._held_dir)
                    else:
                        self._move_piece(self._held_dir)
            else:
                # DAS charged, handle ARR repeats
                if ARR == 0:
                    # Instant snap every frame
                    self._snap_piece(self._held_dir)
                else:
                    self._arr_timer += delta
                    while self._arr_timer >= ARR:
                        self._arr_timer -= ARR
                        self._move_piece(self._held_dir)

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

    def _snap_piece(self, direction: str):
        """Move piece as far as possible in the given direction (ARR=0 behavior)."""
        piece = self.piece
        if direction == "left":
            while self._can_move_to(piece.shape, piece.row, piece.col - 1):
                piece.col -= 1
        elif direction == "right":
            while self._can_move_to(piece.shape, piece.row, piece.col + 1):
                piece.col += 1

    def _get_ghost_row(self) -> int:
        """Get the row where the current piece would land."""
        piece = self.piece
        row = piece.row
        while self._can_move_to(piece.shape, row + 1, piece.col):
            row += 1
        return row

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

        # draw ghost piece
        piece = self.piece
        ghost_row = self._get_ghost_row()
        if ghost_row != piece.row:
            ghost_fill = tuple(c // 4 for c in COLORS[piece.color_id])
            ghost_border = tuple(c // 2 for c in COLORS[piece.color_id])
            for r, c in np.argwhere(piece.shape):
                rect = pygame.Rect(
                    self.x_offset + (piece.col + c) * CELL_SIZE,
                    (ghost_row + r) * CELL_SIZE,
                    CELL_SIZE - 1,
                    CELL_SIZE - 1,
                )
                pygame.draw.rect(screen, ghost_fill, rect)
                pygame.draw.rect(screen, ghost_border, rect, 1)

        # draw current piece
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

    def render_preview(self, screen: pygame.surface.Surface, font: pygame.font.Font, preview_x: int):
        """Render next piece and hold piece preview at the given x position."""
        # HOLD label and box
        hold_label = font.render("HOLD", True, (200, 200, 200))
        screen.blit(hold_label, (preview_x + PREVIEW_W // 2 - hold_label.get_width() // 2, 30))

        hold_box = pygame.Rect(preview_x + 5, 55, PREVIEW_W - 10, PREVIEW_W - 10)
        pygame.draw.rect(screen, (50, 50, 50), hold_box, border_radius=4)
        border_color = (60, 60, 60) if self.hold_used else (80, 80, 80)
        pygame.draw.rect(screen, border_color, hold_box, 1, border_radius=4)

        if self.hold_piece is not None:
            shape, color_id = self.hold_piece
            piece_h, piece_w = shape.shape
            cell = 18
            start_x = preview_x + PREVIEW_W // 2 - (piece_w * cell) // 2
            start_y = 55 + (PREVIEW_W - 10) // 2 - (piece_h * cell) // 2
            color = COLORS[color_id] if not self.hold_used else tuple(c // 2 for c in COLORS[color_id])
            for r, c in np.argwhere(shape):
                rect = pygame.Rect(
                    start_x + c * cell,
                    start_y + r * cell,
                    cell - 1,
                    cell - 1,
                )
                pygame.draw.rect(screen, color, rect)

        # NEXT label and box
        next_label = font.render("NEXT", True, (200, 200, 200))
        next_y = 155
        screen.blit(next_label, (preview_x + PREVIEW_W // 2 - next_label.get_width() // 2, next_y))

        next_box = pygame.Rect(preview_x + 5, next_y + 25, PREVIEW_W - 10, PREVIEW_W - 10)
        pygame.draw.rect(screen, (50, 50, 50), next_box, border_radius=4)
        pygame.draw.rect(screen, (80, 80, 80), next_box, 1, border_radius=4)

        shape, color_id = self._get_next_piece_info()
        piece_h, piece_w = shape.shape
        cell = 18
        start_x = preview_x + PREVIEW_W // 2 - (piece_w * cell) // 2
        start_y = next_y + 25 + (PREVIEW_W - 10) // 2 - (piece_h * cell) // 2

        for r, c in np.argwhere(shape):
            rect = pygame.Rect(
                start_x + c * cell,
                start_y + r * cell,
                cell - 1,
                cell - 1,
            )
            pygame.draw.rect(screen, COLORS[color_id], rect)

    def _get_next_piece_info(self):
        """Peek at next piece in bag without removing it."""
        if not self.bag:
            return self.piece.shape, self.piece.color_id
        return self.bag[-1]

    def _respawn_piece(self) -> Piece:
        """Respawn a new piece once we start the game or place a piece"""
        if len(self.bag) < 7:
            new_bag = TETROMINOS.copy()
            random.shuffle(new_bag)
            self.bag = new_bag + self.bag

        shape, color_id = self.bag.pop()
        return Piece(shape, color_id)

    def _can_move_to(self, shape: np.ndarray, row: int, col: int) -> bool:
        """Check whether we can move shape to (row, col)"""
        for r, c in np.argwhere(shape):
            new_row = row + r
            new_col = col + c
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
        attack_table = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
        garbage_to_send = attack_table.get(cleared_count, 4)
        if garbage_to_send and self.opponent:
            self.opponent.respawn_garbage_lines(garbage_to_send)

        self.piece = self._respawn_piece()
        self.hold_used = False  # reset hold for new piece

        if not self._can_move_to(self.piece.shape, self.piece.row, self.piece.col):
            self.game_over = True

    def _clear_lines(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Clear lines (return number of cleared lines, excluding garbage lines, and resultant board)"""
        complete_lines = []
        garbage_line_count = 0
        for row in range(ROWS):
            if all(self.board[row]):
                complete_lines.append(row)
                if 8 in self.cell_colors[row]:
                    garbage_line_count += 1

        line_cleared = len(complete_lines) - garbage_line_count
        self.normal_lines_cleared += line_cleared
        self.garbage_lines_cleared += garbage_line_count
        self.score += SCORE_TABLE.get(line_cleared, 800)

        remaining_lines = [row for row in range(ROWS) if row not in complete_lines]
        empty_lines = np.zeros((len(complete_lines), COLS), dtype=int)

        new_board = np.vstack((empty_lines.copy(), self.board[remaining_lines]))
        new_cell_colors = np.vstack(
            (empty_lines.copy(), self.cell_colors[remaining_lines])
        )

        return line_cleared, new_board, new_cell_colors

    def _get_heights(self) -> np.ndarray:
        heights = np.zeros(COLS, dtype=int)
        for col in range(COLS):
            for row in range(ROWS):
                if self.board[row][col]:
                    heights[col] = ROWS - row
                    break
        return heights

    def get_game_state(self) -> dict:
        heights = self._get_heights()
        return {
            "heights": heights,
            "max_height": int(heights.max()),
            "bumpiness": self._get_bumpiness(heights),
            "holes": self._get_holes(heights),
        }

    def _get_bumpiness(self, heights: np.ndarray = None) -> int:
        if heights is None:
            heights = self._get_heights()
        bumpiness = 0
        for col in range(COLS - 1):
            bumpiness += abs(heights[col] - heights[col + 1])
        return bumpiness

    def _get_holes(self, heights: np.ndarray = None) -> int:
        if heights is None:
            heights = self._get_heights()
        holes = 0
        for col in range(COLS):
            if heights[col] != 0:
                for row in range(ROWS - heights[col], ROWS):
                    if not self.board[row][col]:
                        holes += 1
        return holes