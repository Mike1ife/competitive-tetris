import pygame
from config import WIN_W, FPS, COLORS, AGENT1_SOURCE, AGENT2_SOURCE


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


def run_home(
    screen: pygame.surface.Surface,
    clock: pygame.time.Clock,
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    last_stats: dict = None,
) -> dict:
    btn_pva = Button((WIN_W // 2 - 160, 160, 320, 52), "Player vs Agent", tag="pva")
    btn_pvp = Button((WIN_W // 2 - 160, 228, 320, 52), "Player vs Player", tag="pvp")
    btn_ava = Button((WIN_W // 2 - 160, 296, 320, 52), "Agent vs Agent", tag="ava")
    btn_pva.selected = True
    mode = "pva"
    mode_btns = [btn_pvp, btn_pva, btn_ava]

    difficulty_btns = {
        "easy": Button((WIN_W // 2 - 160, 224, 96, 38), "Easy", tag="easy"),
        "medium": Button((WIN_W // 2 - 48, 224, 96, 38), "Medium", tag="medium"),
        "hard": Button((WIN_W // 2 + 64, 224, 96, 38), "Hard", tag="hard"),
    }
    difficulty_btns["easy"].selected = True
    difficulty = "easy"

    btn_start = Button((WIN_W // 2 - 100, 380, 200, 48), "Start", tag="start")

    while True:
        clock.tick(FPS)

        offset_y = 50
        if mode == "pva":
            btn_pvp.rect.topleft = (WIN_W // 2 - 160, 228 + offset_y)
            btn_ava.rect.topleft = (WIN_W // 2 - 160, 296 + offset_y)
            btn_start.rect.topleft = (WIN_W // 2 - 100, 380 + offset_y)
            hints_y = 500
            active_btns = mode_btns + list(difficulty_btns.values()) + [btn_start]
        else:
            btn_pvp.rect.topleft = (WIN_W // 2 - 160, 228)
            btn_ava.rect.topleft = (WIN_W // 2 - 160, 296)
            btn_start.rect.topleft = (WIN_W // 2 - 100, 380)
            hints_y = 450
            active_btns = mode_btns + [btn_start]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

            for btn in active_btns:
                if btn.trigger(event):
                    if btn.tag in ("pva", "pvp", "ava"):
                        mode = btn.tag
                        for mode_btn in mode_btns:
                            mode_btn.selected = mode_btn.tag == mode
                    elif btn.tag in difficulty_btns:
                        difficulty = btn.tag
                        for d, difficulty_btn in difficulty_btns.items():
                            difficulty_btn.selected = d == difficulty
                    elif btn.tag == "start":
                        p1_agent_source = None if mode != "ava" else AGENT1_SOURCE
                        p2_agent_source = None if mode == "pvp" else AGENT2_SOURCE
                        return {
                            "p1_agent_source": p1_agent_source,
                            "p2_agent_source": p2_agent_source,
                            "difficulty": difficulty if mode == "pva" else "none",
                        }

        screen.fill(COLORS[0])
        title = font_lg.render("Competitive TETRIS", True, (230, 230, 230))
        screen.blit(title, title.get_rect(center=(WIN_W // 2, 90)))

        tetromino_colors = list(COLORS.values())[1:-1]
        bw = 22
        total = len(tetromino_colors) * (bw + 4) - 4
        x0 = WIN_W // 2 - total // 2
        for i, c in enumerate(tetromino_colors):
            pygame.draw.rect(
                screen, c, (x0 + i * (bw + 4), 118, bw, bw), border_radius=3
            )

        for mode_btn in mode_btns:
            mode_btn.render(screen, font_sm)

        if mode == "pva":
            for difficulty_btn in difficulty_btns.values():
                difficulty_btn.render(screen, font_sm)

        btn_start.render(screen, font_lg)

        hints = [
            "P1: A/D move · W/Z rotate · X 180 · Q hold · S soft · Space hard",
            "P2: ←/→ move · ↑/. rotate · , 180 · RShift hold · ↓ soft · Num0 hard",
        ]
        for i, hint in enumerate(hints):
            s = font_sm.render(hint, True, (140, 140, 140))
            screen.blit(s, s.get_rect(center=(WIN_W // 2, hints_y + i * 22)))

        if last_stats:
            stats_y = hints_y + 60
            winner_color = (255, 255, 100) if last_stats["winner"] else (180, 180, 180)
            winner_text = last_stats["winner"] or "No result"
            s = font_sm.render(f"Last game: {winner_text}", True, winner_color)
            screen.blit(s, s.get_rect(center=(WIN_W // 2, stats_y)))

            p1c = last_stats["p1_clears"]
            p2c = last_stats["p2_clears"]
            stat_lines = [
                f"P1: {last_stats['p1_score']} pts  {last_stats['p1_lines']} lines  T={p1c.get(4,0)} t={p1c.get(3,0)} d={p1c.get(2,0)} s={p1c.get(1,0)}",
                f"P2: {last_stats['p2_score']} pts  {last_stats['p2_lines']} lines  T={p2c.get(4,0)} t={p2c.get(3,0)} d={p2c.get(2,0)} s={p2c.get(1,0)}",
            ]
            for i, line in enumerate(stat_lines):
                s = font_sm.render(line, True, (160, 160, 160))
                screen.blit(s, s.get_rect(center=(WIN_W // 2, stats_y + 22 + i * 20)))

        pygame.display.flip()