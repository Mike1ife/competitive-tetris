"""TODO actual agent"""

import random
from decider import Decider
from pathfinder import Pathfinder


class Agent:
    def __init__(self, source, commands):
        # load policy / weight from source then perform action based on states
        self.commands = commands
        """TODO use decider to get a position, then use pathfinder to output a sequence of commands"""
        self.decider = Decider()
        self.pathfinder = Pathfinder()
        ...

    def random_move(self):
        return random.choice(self.commands)
