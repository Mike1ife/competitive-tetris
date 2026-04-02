from agents.decider import Decider
from agents.pathfinder import Pathfinder


class Agent:
    def __init__(self, source, commands):
        self.commands = commands
        self.decider = Decider(source)
        self.pathfinder = Pathfinder()

    def get_command_sequence(self, board, piece, opp_agg, hold_info=None, hold_used=False, next_piece_info=None):
        actions = self.pathfinder.get_actions(board, piece, hold_info, hold_used, next_piece_info)
        return self.decider.get_sequence(actions, piece, opp_agg, hold_info, hold_used, next_piece_info)

    def get_best_action(self, board, piece, opp_agg, hold_info=None, hold_used=False, next_piece_info=None):
        actions = self.pathfinder.get_actions(board, piece, hold_info, hold_used, next_piece_info)
        return self.decider.get_best_action(actions, piece, opp_agg, hold_info, hold_used, next_piece_info)