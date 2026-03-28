import numpy as np
from tetris import Piece
from models.decider import Decider
from models.pathfinder import Pathfinder


class Agent:
    def __init__(self, source, commands):
        # load policy / weight from source then perform action based on states
        self.commands = commands
        self.decider = Decider(source)
        self.pathfinder = Pathfinder()

    def get_command_sequence(self, board: np.ndarray, piece: Piece, opp_agg: int) -> list:
        actions = self.pathfinder.get_actions(board, piece)
        return self.decider.get_sequence(actions)