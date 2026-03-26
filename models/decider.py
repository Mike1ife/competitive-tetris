import random


class Decider:
    """Model that choose the most optimal legal position to place the piece"""

    def __init__(self): ...

    def get_sequence(self, actions: dict):
        return random.choice(actions)["sequence"]
