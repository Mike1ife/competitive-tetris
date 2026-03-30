from agents.decider import Decider
from agents.pathfinder import Pathfinder


class Agent:
    def __init__(self, source, commands):
        self.commands = commands
        self.decider = Decider(source)
        self.pathfinder = Pathfinder()

    def get_command_sequence(self, board, piece, opp_agg, hold_info=None, hold_used=False):
        actions = self.pathfinder.get_actions(board, piece, hold_info, hold_used)
        return self.decider.get_sequence(actions, piece, opp_agg, hold_info, hold_used)