import argparse
import pygame
from config import (
    WIN_W,
    WIN_H,
    P1_COMMANDS,
    P2_COMMANDS,
    BOARD_W,
    PADDING,
    FPS,
)
from tetris import Tetris
from agent import Agent


class Game:
    def __init__(self, p1_agent_src=None, p2_agent_src=None):
        print(p1_agent_src, p2_agent_src)
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("2-Player Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

        self.p1_agent = self._load_agent(p1_agent_src, list(P1_COMMANDS.keys()))
        self.p2_agent = self._load_agent(p2_agent_src, list(P2_COMMANDS.keys()))
        self.p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
        self.p2 = Tetris(x_offset=BOARD_W + PADDING, commands=P2_COMMANDS)
        self.p1.opponent = self.p2
        self.p2.opponent = self.p1

    def _load_agent(self, source, commands):
        if not source:
            return None

        """TODO load actually agent"""
        return Agent(source, commands)

    def _agent_event(self):
        if self.p1_agent:
            """TODO Agent output a command corresponding to action"""
            command = self.p1_agent.random_move()
            self.p1.execute(command)

        if self.p2_agent:
            """TODO Agent output a command corresponding to action"""
            command = self.p2_agent.random_move()
            self.p2.execute(command)

    def _human_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if not self.p1_agent:
                self.p1.handle_event(event)
            if not self.p2_agent:
                self.p2.handle_event(event)

    def run(self):
        while True:
            delta = self.clock.tick(FPS)

            self._agent_event()
            self._human_event()

            self.p1.update(delta)
            self.p2.update(delta)

            self.screen.fill((10, 10, 10))
            self.p1.render(self.screen, self.font)
            self.p2.render(self.screen, self.font)
            pygame.display.flip()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--p1-agent", type=str)
    parse.add_argument("--p2-agent", type=str)
    args = parse.parse_args()
    Game(p1_agent_src=args.p1_agent, p2_agent_src=args.p2_agent).run()
