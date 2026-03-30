from agents.decider import Decider
from agents.pathfinder import Pathfinder


class Agent:
    def __init__(self, source, commands):
        # load policy / weight from source then perform action based on states
        self.commands = commands
        self.decider = Decider(source)
        self.pathfinder = Pathfinder()

    def get_command_sequence(self, board, piece, opp_agg):
        actions = self.pathfinder.get_actions(board, piece)
        return self.decider.get_sequence(actions, piece, opp_agg)
