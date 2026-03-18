import pygame
import numpy as np

AGENT_SOURCE = "agent_source"

CELL_SIZE = 30  # cell size
ROWS, COLS = 20, 10  # row cell num, col cell num
BOARD_W = COLS * CELL_SIZE  # board width
BOARD_H = ROWS * CELL_SIZE  # board height
PADDING = 20  # distance between two boards
WIN_W = BOARD_W * 2 + PADDING  # window width
WIN_H = BOARD_H  # window height
FPS = 60
FALL_INTERVAL = 500  # drop rate = 500 ms
ACCELERATE_INTERVAL = 50  # accelerate rate (pressing down) = 50 ms
MAX_GARBAGE_HOLE = 3  # maximum holes in a garbage line

AGENT_CAPS = {"easy": 7, "medium": 5, "hard": 3, "none": 0}

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
    (np.array([[1, 1, 1, 1]]), 1),  # I
    (np.array([[1, 1], [1, 1]]), 2),  # O
    (np.array([[0, 1, 0], [1, 1, 1]]), 3),  # T
    (np.array([[1, 0], [1, 0], [1, 1]]), 4),  # L
    (np.array([[0, 1], [0, 1], [1, 1]]), 5),  # J
    (np.array([[0, 1, 1], [1, 1, 0]]), 6),  # S
    (np.array([[1, 1, 0], [0, 1, 1]]), 7),  # Z
]

P1_COMMANDS = {
    "left": (pygame.K_a, "a"),
    "right": (pygame.K_d, "d"),
    "down": (pygame.K_s, "s"),
    "rotate": (pygame.K_w, "w"),
    "drop": (pygame.K_SPACE, "p"),
}

P2_COMMANDS = {
    "left": (pygame.K_LEFT, "l"),
    "right": (pygame.K_RIGHT, "r"),
    "down": (pygame.K_DOWN, "n"),
    "rotate": (pygame.K_UP, "u"),
    "drop": (pygame.K_KP_0, "0"),
}
