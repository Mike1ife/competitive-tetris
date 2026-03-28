import pygame
from config import (
    WIN_W,
    WIN_H,
    P1_COMMANDS,
    P2_COMMANDS,
    BOARD_W,
    PADDING,
    FPS,
    AGENT_CAPS,
)
from tetris import Tetris
from models.agent import Agent
from home import run_home


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("2-Player Tetris")
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("monospace", 20, bold=True)

        font_sm = pygame.font.SysFont("monospace", 16)
        options = run_home(self.screen, self.clock, self.font_lg, font_sm)

        """TODO wait cap time before taking action"""
        self.agent_cap = AGENT_CAPS[options["difficulty"]]

        self.p1_agent = self._load_agent(
            options["p1_agent_source"], list(P1_COMMANDS.keys())
        )
        self.p2_agent = self._load_agent(
            options["p2_agent_source"], list(P2_COMMANDS.keys())
        )
        self.p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
        self.p2 = Tetris(x_offset=BOARD_W + PADDING, commands=P2_COMMANDS)
        self.p1.opponent = self.p2
        self.p2.opponent = self.p1

        # pending commands
        self.p1_pending = []
        self.p2_pending = []
        # current handling piece id
        self.p1_piece_id = None
        self.p2_piece_id = None

    def _load_agent(self, source: str, commands: list):
        if not source:
            return None

        """TODO load actually agent"""
        return Agent(source, commands)

    def run(self):
        while True:
            delta = self.clock.tick(FPS)

            self._agent_event()
            self._human_event()

            self.p1.update(delta)
            self.p2.update(delta)

            self.screen.fill((10, 10, 10))
            self.p1.render(self.screen, self.font_lg)
            self.p2.render(self.screen, self.font_lg)
            pygame.display.flip()

    def _agent_event(self):
        if self.p1_agent and not self.p1.game_over:
            current_id = id(self.p1.piece)
            if current_id != self.p1_piece_id:
                self.p1_piece_id = current_id
                self.p1_pending = self.p1_agent.get_command_sequence(
                    self.p1.board.copy(), self.p1.piece, self.p2.get_game_state()["aggregate_height"]
                )

            if self.p1_pending:
                command = self.p1_pending.pop(0)
                print(f"P1 EXECUTE: {command}")
                self.p1.execute(command)

        if self.p2_agent and not self.p2.game_over:
            current_id = id(self.p2.piece)
            if current_id != self.p2_piece_id:
                self.p2_piece_id = current_id
                self.p2_pending = self.p2_agent.get_command_sequence(
                    self.p2.board.copy(), self.p2.piece, self.p1.get_game_state()["aggregate_height"]
                )

            if self.p2_pending:
                command = self.p2_pending.pop(0)
                print(f"P2 EXECUTE: {command}")
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


if __name__ == "__main__":
    Game().run()
