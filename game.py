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


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("2-Player Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

        self.p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
        self.p2 = Tetris(x_offset=BOARD_W + PADDING, commands=P2_COMMANDS)
        self.p1.opponent = self.p2
        self.p2.opponent = self.p1

    def run(self):
        while True:
            delta = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                self.p1.handle_event(event)
                self.p2.handle_event(event)

            self.p1.update(delta)
            self.p2.update(delta)

            self.screen.fill((10, 10, 10))
            self.p1.render(self.screen, self.font)
            self.p2.render(self.screen, self.font)
            pygame.display.flip()


if __name__ == "__main__":
    Game().run()
