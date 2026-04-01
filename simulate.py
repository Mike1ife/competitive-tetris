"""Run a headless 1v1 simulation between two agents."""

import numpy as np
import pygame
from config import P1_COMMANDS, P2_COMMANDS, BOARD_W, PADDING
from tetris import Tetris
from agents.agent import Agent

NUM_GAMES = 100
MAX_TICKS = 20000
MAX_PIECES = 500
P1_MODEL = "tetris_dqn_neutral_v3.keras"
P2_MODEL = "tetris_dqn_offensive_v2.keras"


def run_game(p1_agent, p2_agent):
    p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
    p2 = Tetris(x_offset=BOARD_W + PADDING, commands=P2_COMMANDS)
    p1.opponent = p2
    p2.opponent = p1

    p1_piece = None
    p2_piece = None
    p1_pieces = 0
    p2_pieces = 0
    dt = 1000 // 60

    for _ in range(MAX_TICKS):
        pygame.event.clear()

        if p1.game_over or p2.game_over:
            break
        if p1_pieces >= MAX_PIECES and p2_pieces >= MAX_PIECES:
            break

        if not p1.game_over:
            if p1.piece is not p1_piece:
                p1_piece = p1.piece
                p1_pieces += 1
                if p1_pieces <= MAX_PIECES:
                    cmds = p1_agent.get_command_sequence(
                        p1.board.copy(), p1.piece,
                        p2.get_game_state()["max_height"],
                        p1.hold_piece, p1.hold_used,
                        p1._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p1.execute(cmd)

        if not p2.game_over:
            if p2.piece is not p2_piece:
                p2_piece = p2.piece
                p2_pieces += 1
                if p2_pieces <= MAX_PIECES:
                    cmds = p2_agent.get_command_sequence(
                        p2.board.copy(), p2.piece,
                        p1.get_game_state()["max_height"],
                        p2.hold_piece, p2.hold_used,
                        p2._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p2.execute(cmd)

        p1.update(dt)
        p2.update(dt)

    if p1.game_over and not p2.game_over:
        winner = "p2"
    elif p2.game_over and not p1.game_over:
        winner = "p1"
    elif p1.score > p2.score:
        winner = "p1"
    elif p2.score > p1.score:
        winner = "p2"
    else:
        winner = "draw"

    return {
        "winner": winner,
        "p1_score": p1.score,
        "p2_score": p2.score,
        "p1_lines": p1.normal_lines_cleared,
        "p2_lines": p2.normal_lines_cleared,
        "p1_garbage": p1.garbage_lines_cleared,
        "p2_garbage": p2.garbage_lines_cleared,
        "p1_clears": dict(p1.clear_distribution),
        "p2_clears": dict(p2.clear_distribution),
    }


def main():
    pygame.init()

    p1_agent = Agent(P1_MODEL, list(P1_COMMANDS.keys()))
    p2_agent = Agent(P2_MODEL, list(P2_COMMANDS.keys()))

    p1_wins = 0
    p2_wins = 0
    draws = 0
    p1_scores = []
    p2_scores = []
    p1_lines = []
    p2_lines = []
    p1_garbage = []
    p2_garbage = []
    p1_clears_total = {1: 0, 2: 0, 3: 0, 4: 0}
    p2_clears_total = {1: 0, 2: 0, 3: 0, 4: 0}

    print(f"P1: {P1_MODEL}  vs  P2: {P2_MODEL}")
    print(f"Games: {NUM_GAMES}  Max pieces: {MAX_PIECES}\n")
    print(f"{'Game':>5}  {'Winner':<8}  {'P1 Score':>10}  {'P2 Score':>10}  {'P1 Lines':>10}  {'P2 Lines':>10}")
    print("-" * 62)

    for i in range(1, NUM_GAMES + 1):
        result = run_game(p1_agent, p2_agent)

        if result["winner"] == "p1":
            p1_wins += 1
        elif result["winner"] == "p2":
            p2_wins += 1
        else:
            draws += 1

        p1_scores.append(result["p1_score"])
        p2_scores.append(result["p2_score"])
        p1_lines.append(result["p1_lines"])
        p2_lines.append(result["p2_lines"])
        p1_garbage.append(result["p1_garbage"])
        p2_garbage.append(result["p2_garbage"])
        for k in p1_clears_total:
            p1_clears_total[k] += result["p1_clears"].get(k, 0)
            p2_clears_total[k] += result["p2_clears"].get(k, 0)

        print(
            f"{i:>5}  {result['winner']:<8}"
            f"  {result['p1_score']:>10}  {result['p2_score']:>10}"
            f"  {result['p1_lines']:>10}  {result['p2_lines']:>10}"
        )

    print("\n" + "=" * 62)
    print(f"RESULTS — {NUM_GAMES} Games")
    print("=" * 62)
    print(f"P1 wins: {p1_wins}  P2 wins: {p2_wins}  Draws: {draws}")
    print(f"P1 win rate: {p1_wins / NUM_GAMES * 100:.1f}%  |  P2 win rate: {p2_wins / NUM_GAMES * 100:.1f}%")
    print()
    print(f"{'':14s}  {'P1':>10s}  {'P2':>10s}")
    print(f"{'Avg score':14s}  {np.mean(p1_scores):10.1f}  {np.mean(p2_scores):10.1f}")
    print(f"{'Max score':14s}  {np.max(p1_scores):10d}  {np.max(p2_scores):10d}")
    print(f"{'Avg lines':14s}  {np.mean(p1_lines):10.1f}  {np.mean(p2_lines):10.1f}")
    print(f"{'Max lines':14s}  {np.max(p1_lines):10d}  {np.max(p2_lines):10d}")
    print(f"{'Avg garb clr':14s}  {np.mean(p1_garbage):10.1f}  {np.mean(p2_garbage):10.1f}")
    print()
    print(f"{'Clear dist':14s}  {'P1':>10s}  {'P2':>10s}")
    print(f"{'Singles':14s}  {p1_clears_total[1]:10d}  {p2_clears_total[1]:10d}")
    print(f"{'Doubles':14s}  {p1_clears_total[2]:10d}  {p2_clears_total[2]:10d}")
    print(f"{'Triples':14s}  {p1_clears_total[3]:10d}  {p2_clears_total[3]:10d}")
    print(f"{'Tetrises':14s}  {p1_clears_total[4]:10d}  {p2_clears_total[4]:10d}")

    pygame.quit()


if __name__ == "__main__":
    main()