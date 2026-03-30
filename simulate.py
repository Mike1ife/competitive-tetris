"""Simulate N headless agent vs agent games and print stats."""

import pygame
import numpy as np
from config import ROWS, COLS, P1_COMMANDS, P2_COMMANDS, AGENT_SOURCE, AGENT_SOURCE_2
from tetris import Tetris
from models.agent import Agent

NUM_GAMES = 100
MAX_PIECES = 2000

def simulate():
    pygame.init()
    pygame.display.set_mode((1, 1))

    p1_agent = Agent(AGENT_SOURCE, list(P1_COMMANDS.keys()))
    p2_agent = Agent(AGENT_SOURCE_2, list(P2_COMMANDS.keys()))

    p1_wins = 0
    p2_wins = 0
    draws = 0
    p1_lines = []
    p2_lines = []
    p1_garbage = []
    p2_garbage = []
    p1_scores = []
    p2_scores = []
    p1_pieces = []
    p2_pieces = []
    game_lengths = []

    for game_num in range(1, NUM_GAMES + 1):
        p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
        p2 = Tetris(x_offset=0, commands=P2_COMMANDS)
        p1.opponent = p2
        p2.opponent = p1

        pieces = 0
        while not p1.game_over and not p2.game_over and pieces < MAX_PIECES:
            pieces += 1

            # P1 acts
            if not p1.game_over:
                cmds = p1_agent.get_command_sequence(
                    p1.board.copy(), p1.piece,
                    p2.get_game_state()["aggregate_height"]
                )
                for cmd in cmds:
                    p1.execute(cmd)

            # P2 acts
            if not p2.game_over:
                cmds = p2_agent.get_command_sequence(
                    p2.board.copy(), p2.piece,
                    p1.get_game_state()["aggregate_height"]
                )
                for cmd in cmds:
                    p2.execute(cmd)

            pygame.event.clear()

        if p1.game_over and p2.game_over:
            draws += 1
            result = "Draw"
        elif p1.game_over:
            p2_wins += 1
            result = "P2 wins"
        elif p2.game_over:
            p1_wins += 1
            result = "P1 wins"
        else:
            draws += 1
            result = "Draw (max pieces)"

        p1_lines.append(p1.normal_lines_cleared)
        p2_lines.append(p2.normal_lines_cleared)
        p1_garbage.append(p1.garbage_lines_cleared)
        p2_garbage.append(p2.garbage_lines_cleared)
        p1_scores.append(p1.score)
        p2_scores.append(p2.score)
        game_lengths.append(pieces)

        p1_height = p1.get_game_state()["aggregate_height"]
        p2_height = p2.get_game_state()["aggregate_height"]

        print(
            f"Game {game_num:3d}: {result:16s}  pieces={pieces:4d}  "
            f"P1(lines={p1.normal_lines_cleared:2d}, garb={p1.garbage_lines_cleared}, score={p1.score:5d}, h={p1_height:2d})  "
            f"P2(lines={p2.normal_lines_cleared:2d}, garb={p2.garbage_lines_cleared}, score={p2.score:5d}, h={p2_height:2d})"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Games: {NUM_GAMES}  |  P1 wins: {p1_wins}  P2 wins: {p2_wins}  Draws: {draws}")
    print(f"P1 win rate: {p1_wins / NUM_GAMES * 100:.1f}%  |  P2 win rate: {p2_wins / NUM_GAMES * 100:.1f}%")
    print()
    print(f"{'':12s}  {'P1':>10s}  {'P2':>10s}")
    print(f"{'Avg lines':12s}  {np.mean(p1_lines):10.1f}  {np.mean(p2_lines):10.1f}")
    print(f"{'Max lines':12s}  {np.max(p1_lines):10d}  {np.max(p2_lines):10d}")
    print(f"{'Avg garbage':12s}  {np.mean(p1_garbage):10.1f}  {np.mean(p2_garbage):10.1f}")
    print(f"{'Avg score':12s}  {np.mean(p1_scores):10.1f}  {np.mean(p2_scores):10.1f}")
    print(f"{'Max score':12s}  {np.max(p1_scores):10d}  {np.max(p2_scores):10d}")
    print()
    print(f"Avg game length: {np.mean(game_lengths):.1f} pieces")
    print(f"Longest game:    {np.max(game_lengths)} pieces")
    print(f"Shortest game:   {np.min(game_lengths)} pieces")


if __name__ == "__main__":
    simulate()