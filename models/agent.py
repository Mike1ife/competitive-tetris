import random
import numpy as np
from tetris import Piece
from models.decider import Decider
from models.pathfinder import Pathfinder


class Agent:
    def __init__(self, source, commands):
        # load policy / weight from source then perform action based on states
        self.commands = commands
        """TODO use decider to get a position, then use pathfinder to output a sequence of commands"""
        self.decider = Decider(source)
        self.pathfinder = Pathfinder()

    def random_move(self):
        return random.choice(self.commands)

    def get_command_sequence(self, board: np.ndarray, piece: Piece, opp_agg: int) -> list:
        actions = self.pathfinder.get_actions(board, piece)
        sequence = self.decider.get_sequence(actions, piece, opp_agg)
        return sequence
