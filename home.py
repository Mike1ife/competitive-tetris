import os
import pygame
from config import WIN_W, FPS, COLORS


class Button:
    def __init__(self, rect: tuple, label: str, tag: str = None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.tag = tag
        self.selected = False
        self.hovered = False

        self._BUTTON_COLOR = (45, 45, 45)
        self._SELECTED_COLOR = (100, 88, 210)
        self._HOVERED_COLOR = (120, 120, 120)
        self._BORDER_COLOR = (70, 70, 70)
        self._TEXT_COLOR = (230, 230, 230)

    def render(self, screen: pygame.surface.Surface, font: pygame.font.Font):
        if self.selected:
            bg_color = self._BUTTON_COLOR
            border_color = self._SELECTED_COLOR
        elif self.hovered:
            bg_color = self._BUTTON_COLOR
            border_color = self._HOVERED_COLOR
        else:
            bg_color = self._BUTTON_COLOR
            border_color = self._BORDER_COLOR

        pygame.draw.rect(screen, bg_color, self.rect, border_radius=8)
        pygame.draw.rect(screen, border_color, self.rect, 1, border_radius=8)

        text = font.render(self.label, True, self._TEXT_COLOR)
        screen.blit(text, text.get_rect(center=self.rect.center))

    def trigger(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False


class ModelSelector:
    def __init__(self, center_x: int, y: int, label: str, models: list, index: int = 0):
        self.label = label
        self.models = models
        self.index = index
        self.center_x = center_x
        self.y = y
        self.btn_left = Button((center_x - 190, y, 30, 30), "<<", tag="left")
        self.btn_right = Button((center_x + 160, y, 30, 30), ">>", tag="right")

    @property
    def value(self):
        return self.models[self.index] if self.models else None

    def render(self, screen: pygame.surface.Surface, font: pygame.font.Font):
        label_surf = font.render(self.label, True, (160, 160, 160))
        screen.blit(label_surf, (self.center_x - label_surf.get_width() // 2, self.y - 18))

        self.btn_left.render(screen, font)
        self.btn_right.render(screen, font)

        name = self.value or "—"
        text = font.render(name, True, (230, 230, 230))
        screen.blit(text, (self.center_x - text.get_width() // 2, self.y + 6))

    def trigger(self, event: pygame.event.Event):
        if not self.models:
            return
        if self.btn_left.trigger(event):
            self.index = (self.index - 1) % len(self.models)
        if self.btn_right.trigger(event):
            self.index = (self.index + 1) % len(self.models)


def _get_model_list():
    models_dir = "./models"
    if not os.path.isdir(models_dir):
        return []
    return sorted(f for f in os.listdir(models_dir) if f.endswith(".keras"))


def run_home(
    screen: pygame.surface.Surface,
    clock: pygame.time.Clock,
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    last_stats: dict = None,
    last_agent_indices: tuple = (0, 0),
) -> dict:
    model_list = _get_model_list()
    idx1, idx2 = last_agent_indices

    sel_agent1 = ModelSelector(WIN_W // 2, 120, "Agent 1", model_list, idx1)
    sel_agent2 = ModelSelector(WIN_W // 2, 170, "Agent 2 (vs player)", model_list, idx2)

    btn_pva = Button((WIN_W // 2 - 160, 230, 320, 44), "Player vs Agent", tag="pva")
    btn_pvp = Button((WIN_W // 2 - 160, 286, 320, 44), "Player vs Player", tag="pvp")
    btn_ava = Button((WIN_W // 2 - 160, 342, 320, 44), "Agent vs Agent", tag="ava")
    btn_pva.selected = True
    mode = "pva"
    mode_btns = [btn_pvp, btn_pva, btn_ava]

    difficulty_btns = {
        "easy": Button((WIN_W // 2 - 160, 344, 96, 38), "Easy", tag="easy"),
        "medium": Button((WIN_W // 2 - 48, 344, 96, 38), "Medium", tag="medium"),
        "hard": Button((WIN_W // 2 + 64, 344, 96, 38), "Hard", tag="hard"),
    }
    difficulty_btns["easy"].selected = True
    difficulty = "easy"

    btn_start = Button((WIN_W // 2 - 100, 500, 200, 48), "Start", tag="start")

    while True:
        clock.tick(FPS)

        if mode == "pva":
            btn_pva.rect.topleft = (WIN_W // 2 - 160, 230)
            btn_pvp.rect.topleft = (WIN_W // 2 - 160, 286)
            btn_ava.rect.topleft = (WIN_W // 2 - 160, 342)
            for key, dx in zip(["easy", "medium", "hard"], [0, 112, 224]):
                difficulty_btns[key].rect.topleft = (WIN_W // 2 - 160 + dx, 400)
            btn_start.rect.topleft = (WIN_W // 2 - 100, 452)
            hints_y = 516
        else:
            btn_pva.rect.topleft = (WIN_W // 2 - 160, 230)
            btn_pvp.rect.topleft = (WIN_W // 2 - 160, 286)
            btn_ava.rect.topleft = (WIN_W // 2 - 160, 342)
            btn_start.rect.topleft = (WIN_W // 2 - 100, 400)
            hints_y = 464

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

            sel_agent1.trigger(event)
            sel_agent2.trigger(event)

            for btn in mode_btns:
                if btn.trigger(event):
                    if btn.tag in ("pva", "pvp", "ava"):
                        mode = btn.tag
                        for mode_btn in mode_btns:
                            mode_btn.selected = mode_btn.tag == mode

            if mode == "pva":
                for key, difficulty_btn in difficulty_btns.items():
                    if difficulty_btn.trigger(event):
                        difficulty = key
                        for d, db in difficulty_btns.items():
                            db.selected = d == difficulty

            if btn_start.trigger(event):
                if mode == "ava":
                    p1_src = sel_agent1.value
                    p2_src = sel_agent2.value
                elif mode == "pva":
                    p1_src = None
                    p2_src = sel_agent2.value
                else:
                    p1_src = None
                    p2_src = None
                return {
                    "p1_agent_source": p1_src,
                    "p2_agent_source": p2_src,
                    "difficulty": difficulty if mode == "pva" else "none",
                    "agent_indices": (sel_agent1.index, sel_agent2.index),
                }

        screen.fill(COLORS[0])
        title = font_lg.render("Competitive TETRIS", True, (230, 230, 230))
        screen.blit(title, title.get_rect(center=(WIN_W // 2, 40)))

        tetromino_colors = list(COLORS.values())[1:-1]
        bw = 22
        total = len(tetromino_colors) * (bw + 4) - 4
        x0 = WIN_W // 2 - total // 2
        for i, c in enumerate(tetromino_colors):
            pygame.draw.rect(
                screen, c, (x0 + i * (bw + 4), 62, bw, bw), border_radius=3
            )

        sel_agent1.render(screen, font_sm)
        sel_agent2.render(screen, font_sm)

        for mode_btn in mode_btns:
            mode_btn.render(screen, font_sm)

        if mode == "pva":
            for difficulty_btn in difficulty_btns.values():
                difficulty_btn.render(screen, font_sm)

        btn_start.render(screen, font_lg)

        def render_key_hint(screen, font, x, y, label, label_color, keys):
            """Render a control hint with highlighted key backgrounds."""
            lbl = font.render(label, True, label_color)
            screen.blit(lbl, (x, y))
            cx = x + lbl.get_width() + 10
            for part in keys:
                if part.startswith("["):
                    key_text = part[1:-1]
                    key_surf = font.render(key_text, True, (255, 255, 255))
                    pad = 4
                    bg_rect = pygame.Rect(cx - pad, y - 2, key_surf.get_width() + pad * 2, key_surf.get_height() + 4)
                    pygame.draw.rect(screen, (70, 65, 120), bg_rect, border_radius=4)
                    pygame.draw.rect(screen, (100, 90, 160), bg_rect, 1, border_radius=4)
                    screen.blit(key_surf, (cx, y))
                    cx += key_surf.get_width() + pad * 2 + 4
                else:
                    txt = font.render(part, True, (180, 180, 180))
                    screen.blit(txt, (cx, y))
                    cx += txt.get_width() + 4

        p1_keys = ["[A]","[D]","move  ","[W]","[Z]","rot  ","[X]","180  ","[Q]","hold  ","[S]","soft  ","[Space]","hard"]
        p2_keys = ["[\u2190]","[\u2192]","move  ","[\u2191]","[.]","rot  ","[,]","180  ","[RShift]","hold  ","[\u2193]","soft  ","[/]","hard"]

        total_test = font_sm.render("A D move W Z rot X 180 Q hold S soft Space hard      ", True, (0,0,0))
        p1_x = WIN_W // 2 - total_test.get_width() // 2 - 20
        render_key_hint(screen, font_sm, p1_x, hints_y, "P1:", (100, 200, 255), p1_keys)
        render_key_hint(screen, font_sm, p1_x, hints_y + 30, "P2:", (255, 180, 100), p2_keys)

        if last_stats:
            stats_y = hints_y + 60
            winner_color = (255, 255, 100) if last_stats["winner"] else (200, 200, 200)
            winner_text = last_stats["winner"] or "No result"
            s = font_sm.render(f"Last game: {winner_text}", True, winner_color)
            screen.blit(s, s.get_rect(center=(WIN_W // 2, stats_y)))

            p1c = last_stats["p1_clears"]
            p2c = last_stats["p2_clears"]
            stat_lines = [
                f"P1: {last_stats['p1_score']} pts  {last_stats['p1_lines']} lines  g={last_stats.get('p1_garbage_sent', 0)}  T={p1c.get(4,0)} t={p1c.get(3,0)} d={p1c.get(2,0)} s={p1c.get(1,0)}  c={last_stats.get('p1_combos', 0)}",
                f"P2: {last_stats['p2_score']} pts  {last_stats['p2_lines']} lines  g={last_stats.get('p2_garbage_sent', 0)}  T={p2c.get(4,0)} t={p2c.get(3,0)} d={p2c.get(2,0)} s={p2c.get(1,0)}  c={last_stats.get('p2_combos', 0)}",
            ]
            for i, line in enumerate(stat_lines):
                s = font_sm.render(line, True, (210, 210, 210))
                screen.blit(s, s.get_rect(center=(WIN_W // 2, stats_y + 22 + i * 20)))

        pygame.display.flip()