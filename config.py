import pygame
import numpy as np

CELL_SIZE = 30  # cell size
ROWS, COLS = 20, 10  # row cell num, col cell num
BOARD_W = COLS * CELL_SIZE  # board width
BOARD_H = ROWS * CELL_SIZE  # board height
PADDING = 20  # distance between two boards
PREVIEW_W = 80  # width of next piece preview panel
WIN_W = BOARD_W * 2 + PADDING + PREVIEW_W * 2  # window width (preview on each side)
WIN_H = BOARD_H + 60  # window height
FPS = 60
FALL_INTERVAL = 500  # drop rate = 500 ms
ACCELERATE_INTERVAL = 50  # accelerate rate (pressing down) = 50 ms

# DAS/ARR for left/right movement (ms)
DAS = 167  # delay before auto-shift starts
ARR = 33  # interval between repeated moves once auto-shift active (0 = instant)

AGENT_CAPS = {"easy": 1000, "medium": 500, "hard": 333, "none": 100}

COLORS = {
    0: (30, 30, 30),  # Background
    1: (0, 240, 240),  # I - cyan
    2: (240, 240, 0),  # O - yellow
    3: (160, 0, 240),  # T - purple
    4: (240, 160, 0),  # L - orange
    5: (0, 0, 240),  # J - blue
    6: (0, 240, 0),  # S - green
    7: (240, 0, 0),  # Z - red
    8: (180, 180, 180),  # Garbage
}

# (shape, color)
TETROMINOS = [
    (np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]), 1),  # I
    (np.array([[1, 1], [1, 1]]), 2),  # O
    (np.array([[0, 1, 0], [1, 1, 1]]), 3),  # T
    (np.array([[0, 0, 1], [1, 1, 1]]), 4),  # L
    (np.array([[1, 0, 0], [1, 1, 1]]), 5),  # J
    (np.array([[0, 1, 1], [1, 1, 0]]), 6),  # S
    (np.array([[1, 1, 0], [0, 1, 1]]), 7),  # Z
]

TETROMINOS_NAMES = ["", "I", "O", "T", "L", "J", "S", "Z"]

# (current rot id, next rot id)

JLSTZ_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
}

I_KICKS = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
}

O_KICKS = {
    (0, 1): [(0, 0)],
    (1, 2): [(0, 0)],
    (2, 3): [(0, 0)],
    (3, 0): [(0, 0)],
}

WALL_KICKS = {
    "J": JLSTZ_KICKS,
    "L": JLSTZ_KICKS,
    "S": JLSTZ_KICKS,
    "T": JLSTZ_KICKS,
    "Z": JLSTZ_KICKS,
    "I": I_KICKS,
    "O": O_KICKS,
}

P1_COMMANDS = {
    "left": (pygame.K_a, "a"),
    "right": (pygame.K_d, "d"),
    "down": (pygame.K_s, "s"),
    "rotate": (pygame.K_w, "w"),
    "rotate_left": (pygame.K_z, "z"),
    "rotate_180": (pygame.K_x, "x"),
    "hold": (pygame.K_q, "q"),
    "drop": (pygame.K_SPACE, "p"),
}

P2_COMMANDS = {
    "left": (pygame.K_LEFT, "l"),
    "right": (pygame.K_RIGHT, "r"),
    "down": (pygame.K_DOWN, "n"),
    "rotate": (pygame.K_UP, "u"),
    "rotate_left": (pygame.K_PERIOD, "."),
    "rotate_180": (pygame.K_COMMA, ","),
    "hold": (pygame.K_RSHIFT, "/"),
    "drop": (pygame.K_SLASH, "/"),
}

SCORE_TABLE = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}