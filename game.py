import pygame
import numpy as np
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
from agents.agent import Agent
from home import run_home

AVA_GAMES = 50


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
        self.is_ava = options["p1_agent_source"] is not None and options["p2_agent_source"] is not None

        self.p1_agent = self._load_agent(
            options["p1_agent_source"], list(P1_COMMANDS.keys())
        )
        self.p2_agent = self._load_agent(
            options["p2_agent_source"], list(P2_COMMANDS.keys())
        )

        self.p1 = None
        self.p2 = None
        self._init_boards()

        self.p1_pending = []
        self.p2_pending = []
        self.p1_piece_id = None
        self.p2_piece_id = None
        self._p1_last_piece_time = 0
        self._p2_last_piece_time = 0
        self._game_ended = False

        # AvA stats
        self._ava_game_num = 0
        self._ava_p1_wins = 0
        self._ava_p2_wins = 0
        self._ava_draws = 0
        self._ava_p1_lines = []
        self._ava_p2_lines = []
        self._ava_p1_garbage = []
        self._ava_p2_garbage = []
        self._ava_p1_scores = []
        self._ava_p2_scores = []
        self._ava_done = False

    def _init_boards(self):
        self.p1 = Tetris(x_offset=PREVIEW_W, commands=P1_COMMANDS)
        self.p2 = Tetris(x_offset=PREVIEW_W + BOARD_W + PADDING, commands=P2_COMMANDS)
        self.p1.opponent = self.p2
        self.p2.opponent = self.p1

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

            if self.is_ava and not self._ava_done:
                counter_text = self.font_lg.render(
                    f"Game {self._ava_game_num + 1}/{AVA_GAMES}  P1:{self._ava_p1_wins} P2:{self._ava_p2_wins}",
                    True, (255, 255, 100)
                )
                self.screen.blit(counter_text, (WIN_W // 2 - counter_text.get_width() // 2, WIN_H - 30))

            pygame.display.flip()

            if self._game_ended:
                if self._ava_done:
                    continue
                if self.is_ava:
                    self._handle_ava_end()
                else:
                    continue

            if self.p1.game_over and self.p2.game_over:
                winner = "Draw"
            elif self.p1.game_over:
                winner = "P2 wins"
            elif self.p2.game_over:
                winner = "P1 wins"
            else:
                continue

            if self.is_ava:
                self._record_ava_result(winner)
            else:
                print(f"{winner}  p1_score={self.p1.score}  p2_score={self.p2.score}")
                print(f"  P1: lines={self.p1.normal_lines_cleared}  garbage_cleared={self.p1.garbage_lines_cleared}")
                print(f"  P2: lines={self.p2.normal_lines_cleared}  garbage_cleared={self.p2.garbage_lines_cleared}")
                self._game_ended = True
                pygame.time.wait(3000)
                break

    def _record_ava_result(self, winner):
        self._ava_game_num += 1

        if winner == "P1 wins":
            self._ava_p1_wins += 1
        elif winner == "P2 wins":
            self._ava_p2_wins += 1
        else:
            self._ava_draws += 1

        self._ava_p1_lines.append(self.p1.normal_lines_cleared)
        self._ava_p2_lines.append(self.p2.normal_lines_cleared)
        self._ava_p1_garbage.append(self.p1.garbage_lines_cleared)
        self._ava_p2_garbage.append(self.p2.garbage_lines_cleared)
        self._ava_p1_scores.append(self.p1.score)
        self._ava_p2_scores.append(self.p2.score)

        print(
            f"Game {self._ava_game_num:3d}: {winner:10s}  "
            f"P1(lines={self.p1.normal_lines_cleared:2d}, garb={self.p1.garbage_lines_cleared}, score={self.p1.score:5d})  "
            f"P2(lines={self.p2.normal_lines_cleared:2d}, garb={self.p2.garbage_lines_cleared}, score={self.p2.score:5d})"
        )

        if self._ava_game_num >= AVA_GAMES:
            self._print_ava_summary()
            self._game_ended = True
            self._ava_done = True
        else:
            self._init_boards()
            self.p1_piece_id = None
            self.p2_piece_id = None
            self._p1_last_piece_time = 0
            self._p2_last_piece_time = 0

    def _handle_ava_end(self):
        self.screen.fill((10, 10, 10))
        lines = [
            f"Agent vs Agent - {AVA_GAMES} Games",
            "",
            f"P1 wins: {self._ava_p1_wins}   P2 wins: {self._ava_p2_wins}   Draws: {self._ava_draws}",
            f"P1 win rate: {self._ava_p1_wins / AVA_GAMES * 100:.1f}%",
            "",
            f"P1 avg lines: {np.mean(self._ava_p1_lines):.1f}   P2 avg lines: {np.mean(self._ava_p2_lines):.1f}",
            f"P1 avg garb:  {np.mean(self._ava_p1_garbage):.1f}   P2 avg garb:  {np.mean(self._ava_p2_garbage):.1f}",
            f"P1 avg score: {np.mean(self._ava_p1_scores):.0f}   P2 avg score: {np.mean(self._ava_p2_scores):.0f}",
        ]
        y = 150
        for line in lines:
            text = self.font_lg.render(line, True, (230, 230, 230))
            self.screen.blit(text, (WIN_W // 2 - text.get_width() // 2, y))
            y += 30
        pygame.display.flip()

    def _print_ava_summary(self):
        print("\n" + "=" * 60)
        print(f"AGENT VS AGENT - {AVA_GAMES} GAMES")
        print("=" * 60)
        print(f"P1 wins: {self._ava_p1_wins}  P2 wins: {self._ava_p2_wins}  Draws: {self._ava_draws}")
        print(f"P1 win rate: {self._ava_p1_wins / AVA_GAMES * 100:.1f}%  |  P2 win rate: {self._ava_p2_wins / AVA_GAMES * 100:.1f}%")
        print()
        print(f"{'':12s}  {'P1':>10s}  {'P2':>10s}")
        print(f"{'Avg lines':12s}  {np.mean(self._ava_p1_lines):10.1f}  {np.mean(self._ava_p2_lines):10.1f}")
        print(f"{'Max lines':12s}  {np.max(self._ava_p1_lines):10d}  {np.max(self._ava_p2_lines):10d}")
        print(f"{'Avg garbage':12s}  {np.mean(self._ava_p1_garbage):10.1f}  {np.mean(self._ava_p2_garbage):10.1f}")
        print(f"{'Avg score':12s}  {np.mean(self._ava_p1_scores):10.1f}  {np.mean(self._ava_p2_scores):10.1f}")
        print(f"{'Max score':12s}  {np.max(self._ava_p1_scores):10d}  {np.max(self._ava_p2_scores):10d}")

    def _agent_event(self):
        now = pygame.time.get_ticks()

        if self.p1_agent and not self.p1.game_over:
            current_id = id(self.p1.piece)
            if current_id != self.p1_piece_id and (
                now - self._p1_last_piece_time >= self.agent_cap
            ):
                self.p1_piece_id = current_id
                self.p1_pending = self.p1_agent.get_command_sequence(
                    self.p1.board.copy(),
                    self.p1.piece,
                    self.p2.get_game_state()["max_height"],
                    self.p1.hold_piece,
                    self.p1.hold_used,
                )
                self._p1_last_piece_time = now
                for cmd in self.p1_pending:
                    self.p1.execute(cmd)
                self.p1_pending = []

        if self.p2_agent and not self.p2.game_over:
            current_id = id(self.p2.piece)
            if current_id != self.p2_piece_id and (
                now - self._p2_last_piece_time >= self.agent_cap
            ):
                self.p2_piece_id = current_id
                self.p2_pending = self.p2_agent.get_command_sequence(
                    self.p2.board.copy(),
                    self.p2.piece,
                    self.p1.get_game_state()["max_height"],
                    self.p2.hold_piece,
                    self.p2.hold_used,
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