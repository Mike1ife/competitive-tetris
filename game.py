import pygame
from config import (
    WIN_W,
    WIN_H,
    P1_COMMANDS,
    P2_COMMANDS,
    BOARD_W,
    PADDING,
    PREVIEW_W,
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

        self.agent_cap = AGENT_CAPS[options["difficulty"]]

        self.p1_agent = self._load_agent(
            options["p1_agent_source"], list(P1_COMMANDS.keys())
        )
        self.p2_agent = self._load_agent(
            options["p2_agent_source"], list(P2_COMMANDS.keys())
        )
        self.p1 = Tetris(x_offset=PREVIEW_W, commands=P1_COMMANDS)
        self.p2 = Tetris(x_offset=PREVIEW_W + BOARD_W + PADDING, commands=P2_COMMANDS)
        self.p1.opponent = self.p2
        self.p2.opponent = self.p1

        self.p1_pending = []
        self.p2_pending = []
        self.p1_piece_id = None
        self.p2_piece_id = None
        self._p1_last_piece_time = 0
        self._p2_last_piece_time = 0
        self._game_ended = False

    def _load_agent(self, source: str, commands: list):
        if not source:
            return None
        return Agent(source, commands)

    def run(self):
        while True:
            self.clock.tick(FPS)

            self._agent_event()
            self._human_event()

            self.p1.update(self.clock.get_time())
            self.p2.update(self.clock.get_time())

            self.screen.fill((10, 10, 10))
            self.p1.render(self.screen, self.font_lg)
            self.p2.render(self.screen, self.font_lg)
            self.p1.render_preview(self.screen, self.font_lg, 0)
            self.p2.render_preview(self.screen, self.font_lg, PREVIEW_W + BOARD_W * 2 + PADDING)
            pygame.display.flip()

            if self._game_ended:
                continue

            if self.p1.game_over and self.p2.game_over:
                winner = "Draw"
            elif self.p1.game_over:
                winner = "P2 wins"
            elif self.p2.game_over:
                winner = "P1 wins"
            else:
                continue

            print(f"{winner}  p1_score={self.p1.score}  p2_score={self.p2.score}")
            print(f"  P1: lines={self.p1.normal_lines_cleared}  garbage_cleared={self.p1.garbage_lines_cleared}")
            print(f"  P2: lines={self.p2.normal_lines_cleared}  garbage_cleared={self.p2.garbage_lines_cleared}")
            self._game_ended = True
            pygame.time.wait(3000)
            break

    def _agent_event(self):
        now = pygame.time.get_ticks()

        if self.p1_agent and not self.p1.game_over:
            current_id = id(self.p1.piece)
            if current_id != self.p1_piece_id and (now - self._p1_last_piece_time >= self.agent_cap):
                self.p1_piece_id = current_id
                self.p1_pending = self.p1_agent.get_command_sequence(
                    self.p1.board.copy(), self.p1.piece, self.p2.get_game_state()["aggregate_height"]
                )
                self._p1_last_piece_time = now
                for cmd in self.p1_pending:
                    self.p1.execute(cmd)
                self.p1_pending = []

        if self.p2_agent and not self.p2.game_over:
            current_id = id(self.p2.piece)
            if current_id != self.p2_piece_id and (now - self._p2_last_piece_time >= self.agent_cap):
                self.p2_piece_id = current_id
                self.p2_pending = self.p2_agent.get_command_sequence(
                    self.p2.board.copy(), self.p2.piece, self.p1.get_game_state()["aggregate_height"]
                )
                self._p2_last_piece_time = now
                for cmd in self.p2_pending:
                    self.p2.execute(cmd)
                self.p2_pending = []

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