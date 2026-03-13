"""TODO actual agent"""

import random


class Agent:
    def __init__(self, source, commands):
        # load policy / weight from source then perform action based on states
        self.commands = commands
        ...

    def random_move(self):
        return random.choice(self.commands)
