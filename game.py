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
    COLORS,
    CELL_SIZE,
)
from tetris import Tetris
from agents.agent import Agent
from home import run_home


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("2-Player Tetris")
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 16)
        self._run_loop()

    def _run_loop(self):
        last_stats = None
        last_agent_indices = (0, 0)
        while True:
            options = run_home(self.screen, self.clock, self.font_lg, self.font_sm, last_stats, last_agent_indices)
            if options is None:
                break

            last_agent_indices = options.get("agent_indices", (0, 0))
            self.agent_cap = AGENT_CAPS[options["difficulty"]]
            self.is_ava = options["p1_agent_source"] is not None and options["p2_agent_source"] is not None

            self.p1_agent_source = options["p1_agent_source"]
            self.p2_agent_source = options["p2_agent_source"]
            self.p1_agent = self._load_agent(self.p1_agent_source, list(P1_COMMANDS.keys()))
            self.p2_agent = self._load_agent(self.p2_agent_source, list(P2_COMMANDS.keys()))

            result, stats = self._run_game()
            if stats:
                last_stats = stats
            if result == "quit":
                break

    def _get_stats(self, winner=None):
        return {
            "winner": winner,
            "p1_score": self.p1.score,
            "p2_score": self.p2.score,
            "p1_lines": self.p1.normal_lines_cleared,
            "p2_lines": self.p2.normal_lines_cleared,
            "p1_garbage": self.p1.garbage_lines_cleared,
            "p2_garbage": self.p2.garbage_lines_cleared,
            "p1_clears": dict(self.p1.clear_distribution),
            "p2_clears": dict(self.p2.clear_distribution),
            "p1_combos": self._p1_combo_count,
            "p2_combos": self._p2_combo_count,
            "p1_garbage_sent": self.p1.total_garbage_sent,
            "p2_garbage_sent": self.p2.total_garbage_sent,
        }

    def _run_game(self):
        self.p1 = Tetris(x_offset=PREVIEW_W, commands=P1_COMMANDS)
        self.p2 = Tetris(x_offset=PREVIEW_W + BOARD_W + PADDING, commands=P2_COMMANDS)
        self.p1.opponent = self.p2
        self.p2.opponent = self.p1

        if self.p1_agent:
            self.p1.transparent_ghost = True
        if self.p2_agent:
            self.p2.transparent_ghost = True

        self._p1_piece = None
        self._p2_piece = None
        self._p1_last_piece_time = 0
        self._p2_last_piece_time = 0
        self._p1_combo_count = 0
        self._p2_combo_count = 0
        self._p1_max_combo = 0
        self._p2_max_combo = 0
        self._p1_prev_combo = -1
        self._p2_prev_combo = -1
        self._p1_planned = None
        self._p2_planned = None
        self._p1_planned_pending = None
        self._p2_planned_pending = None
        self._p1_cmds = None
        self._p2_cmds = None

        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._print_stats()
                    return "quit", self._get_stats()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    result = self._pause_menu()
                    if result == "quit":
                        self._print_stats()
                        return "quit", self._get_stats()
                    elif result == "menu":
                        self._print_stats()
                        return "menu", self._get_stats()

                if not self.p1_agent:
                    self.p1.handle_event(event)
                if not self.p2_agent:
                    self.p2.handle_event(event)

            self._p1_prev_combo = self.p1.combo
            self._p2_prev_combo = self.p2.combo

            self._agent_event()

            if self.p1.combo > self._p1_prev_combo:
                self._p1_combo_count += 1
            if self.p2.combo > self._p2_prev_combo:
                self._p2_combo_count += 1
            self._p1_max_combo = max(self._p1_max_combo, self.p1.combo)
            self._p2_max_combo = max(self._p2_max_combo, self.p2.combo)

            now = pygame.time.get_ticks()
            if self._p1_planned and not self._p1_cmds:
                self._p1_planned = None
                self.p1.hide_ghost = False
            if self._p2_planned and not self._p2_cmds:
                self._p2_planned = None
                self.p2.hide_ghost = False

            self.p1.update(self.clock.get_time())
            self.p2.update(self.clock.get_time())

            self.screen.fill((10, 10, 10))
            self.p1.render(self.screen, self.font_lg)
            self.p2.render(self.screen, self.font_lg)
            self._render_planned(self.screen, self.p1, self._p1_planned)
            self._render_planned(self.screen, self.p2, self._p2_planned)
            self.p1.render_preview(self.screen, self.font_lg, 0)
            self.p2.render_preview(self.screen, self.font_lg, PREVIEW_W + BOARD_W * 2 + PADDING)
            self._render_model_labels()
            pygame.display.flip()

            if self.p1.game_over or self.p2.game_over:
                if self.p1.game_over and self.p2.game_over:
                    winner = "Draw"
                elif self.p1.game_over:
                    winner = "P2 wins"
                else:
                    winner = "P1 wins"

                self._print_model_info()
                print(f"{winner}  p1_score={self.p1.score}  p2_score={self.p2.score}")
                print(f"  P1: lines={self.p1.normal_lines_cleared + self.p1.garbage_lines_cleared}  garbage_sent={self.p1.total_garbage_sent}  clears={self.p1.clear_distribution}  combos={self._p1_combo_count}  max_combo={self._p1_max_combo}")
                print(f"  P2: lines={self.p2.normal_lines_cleared + self.p2.garbage_lines_cleared}  garbage_sent={self.p2.total_garbage_sent}  clears={self.p2.clear_distribution}  combos={self._p2_combo_count}  max_combo={self._p2_max_combo}")

                stats = self._get_stats(winner)
                result = self._wait_for_back(winner)
                return result, stats

    def _wait_for_back(self, winner):
        cooldown = pygame.time.get_ticks() + 500

        while True:
            self.clock.tick(FPS)

            self.screen.fill((10, 10, 10))
            self.p1.render(self.screen, self.font_lg)
            self.p2.render(self.screen, self.font_lg)
            self.p1.render_preview(self.screen, self.font_lg, 0)
            self.p2.render_preview(self.screen, self.font_lg, PREVIEW_W + BOARD_W * 2 + PADDING)

            msg = self.font_lg.render(winner, True, (255, 255, 100))
            self.screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2 - 40))

            hint = self.font_sm.render("Press any key to return to menu", True, (140, 140, 140))
            self.screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, WIN_H // 2))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if pygame.time.get_ticks() >= cooldown:
                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        return "menu"

    def _pause_menu(self):
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))

        while True:
            self.clock.tick(FPS)

            self.screen.blit(overlay, (0, 0))

            title = self.font_lg.render("PAUSED", True, (255, 255, 100))
            self.screen.blit(title, (WIN_W // 2 - title.get_width() // 2, WIN_H // 2 - 60))

            hints = [
                "Escape - Resume",
                "M - Back to Menu",
                "Q - Quit",
            ]
            for i, hint in enumerate(hints):
                text = self.font_sm.render(hint, True, (180, 180, 180))
                self.screen.blit(text, (WIN_W // 2 - text.get_width() // 2, WIN_H // 2 - 10 + i * 28))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "resume"
                    elif event.key == pygame.K_m:
                        return "menu"
                    elif event.key == pygame.K_q:
                        return "quit"

    def _print_model_info(self):
        if self.p1_agent_source and self.p2_agent_source:
            print(f"P1: {self.p1_agent_source}  vs  P2: {self.p2_agent_source}")
        elif self.p2_agent_source:
            print(f"Agent: {self.p2_agent_source}")

    def _print_stats(self):
        print(f"\n--- Force Quit ---")
        self._print_model_info()
        print(f"  P1: score={self.p1.score}  lines={self.p1.normal_lines_cleared + self.p1.garbage_lines_cleared}  garbage_sent={self.p1.total_garbage_sent}  clears={self.p1.clear_distribution}  combos={self._p1_combo_count}  max_combo={self._p1_max_combo}")
        print(f"  P2: score={self.p2.score}  lines={self.p2.normal_lines_cleared + self.p2.garbage_lines_cleared}  garbage_sent={self.p2.total_garbage_sent}  clears={self.p2.clear_distribution}  combos={self._p2_combo_count}  max_combo={self._p2_max_combo}")

    def _render_model_labels(self):
        from home import _parse_model_name
        y = WIN_H - 40
        if self.p1_agent_source:
            label = self.font_sm.render(_parse_model_name(self.p1_agent_source), True, (160, 160, 160))
            self.screen.blit(label, (PREVIEW_W + (BOARD_W - label.get_width()) // 2, y))
        if self.p2_agent_source:
            label = self.font_sm.render(_parse_model_name(self.p2_agent_source), True, (160, 160, 160))
            self.screen.blit(label, (PREVIEW_W + BOARD_W + PADDING + (BOARD_W - label.get_width()) // 2, y))

    def _render_planned(self, screen, tetris_board, planned):
        """Render a ghost piece showing the agent's planned placement."""
        if planned is None:
            return
        row, col, shape, color_id = planned
        color = COLORS[color_id]
        for r, c in np.argwhere(shape):
            surf = pygame.Surface((CELL_SIZE - 1, CELL_SIZE - 1), pygame.SRCALPHA)
            surf.fill((*color, 20))
            screen.blit(surf, (
                tetris_board.x_offset + (col + c) * CELL_SIZE,
                (row + r) * CELL_SIZE,
            ))

    def _load_agent(self, source: str, commands: list):
        if not source:
            return None
        return Agent(source, commands)

    def _agent_event(self):
        now = pygame.time.get_ticks()

        if self.p1_agent and not self.p1.game_over:
            if self.p1.piece is not self._p1_piece:
                self._p1_piece = self.p1.piece
                self._p1_last_piece_time = now
                action = self.p1_agent.get_best_action(
                    self.p1.board.copy(), self.p1.piece,
                    self.p2.get_game_state()["max_height"],
                    self.p1.hold_piece, self.p1.hold_used,
                    self.p1._get_next_piece_info(),
                )
                if action:
                    self._p1_cmds = action["sequence"]
                    color_id = action["color_id"] + 1
                    if self._p1_cmds and self._p1_cmds[0] == "hold":
                        self.p1.execute("hold")
                        self._p1_cmds = self._p1_cmds[1:]
                        self._p1_piece = self.p1.piece
                        self._p1_last_piece_time = now
                        self.p1.hide_ghost = False
                    self._p1_planned_pending = (action["row"], action["col"], action["shape"], color_id)
                    self._p1_planned = None
                else:
                    self._p1_cmds = []
                    self._p1_planned = None
            if self._p1_cmds and now - self._p1_last_piece_time >= self.agent_cap:
                for cmd in self._p1_cmds:
                    self.p1.execute(cmd)
                self._p1_cmds = None
            elif self._p1_planned_pending and not self._p1_planned and now - self._p1_last_piece_time >= self.agent_cap // 4:
                self._p1_planned = self._p1_planned_pending
                self._p1_planned_pending = None
                self.p1.hide_ghost = True

        if self.p2_agent and not self.p2.game_over:
            if self.p2.piece is not self._p2_piece:
                self._p2_piece = self.p2.piece
                self._p2_last_piece_time = now
                action = self.p2_agent.get_best_action(
                    self.p2.board.copy(), self.p2.piece,
                    self.p1.get_game_state()["max_height"],
                    self.p2.hold_piece, self.p2.hold_used,
                    self.p2._get_next_piece_info(),
                )
                if action:
                    self._p2_cmds = action["sequence"]
                    color_id = action["color_id"] + 1
                    if self._p2_cmds and self._p2_cmds[0] == "hold":
                        self.p2.execute("hold")
                        self._p2_cmds = self._p2_cmds[1:]
                        self._p2_piece = self.p2.piece
                        self._p2_last_piece_time = now
                        self.p2.hide_ghost = False
                    self._p2_planned_pending = (action["row"], action["col"], action["shape"], color_id)
                    self._p2_planned = None
                else:
                    self._p2_cmds = []
                    self._p2_planned = None
            if self._p2_cmds and now - self._p2_last_piece_time >= self.agent_cap:
                for cmd in self._p2_cmds:
                    self.p2.execute(cmd)
                self._p2_cmds = None
            elif self._p2_planned_pending and not self._p2_planned and now - self._p2_last_piece_time >= self.agent_cap // 4:
                self._p2_planned = self._p2_planned_pending
                self._p2_planned_pending = None
                self.p2.hide_ghost = True


if __name__ == "__main__":
    Game()