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

    p1_died = p1.game_over and not p2.game_over
    p2_died = p2.game_over and not p1.game_over

    if p1_died:
        winner = "p2"
        cause = "p1_died"
    elif p2_died:
        winner = "p1"
        cause = "p2_died"
    elif p1.score > p2.score:
        winner = "p1"
        cause = "score"
    elif p2.score > p1.score:
        winner = "p2"
        cause = "score"
    else:
        winner = "draw"
        cause = "draw"

    return {
        "winner": winner,
        "cause": cause,
        "p1_score": p1.score,
        "p2_score": p2.score,
        "p1_lines": p1.normal_lines_cleared,
        "p2_lines": p2.normal_lines_cleared,
        "p1_garbage_cleared": p1.garbage_lines_cleared,
        "p2_garbage_cleared": p2.garbage_lines_cleared,
        "p1_pieces": p1_pieces,
        "p2_pieces": p2_pieces,
        "p1_clears": dict(p1.clear_distribution),
        "p2_clears": dict(p2.clear_distribution),
        "p1_height": p1.get_game_state()["max_height"],
        "p2_height": p2.get_game_state()["max_height"],
    }


def main():
    pygame.init()

    p1_agent = Agent(P1_MODEL, list(P1_COMMANDS.keys()))
    p2_agent = Agent(P2_MODEL, list(P2_COMMANDS.keys()))

    p1_wins = 0
    p2_wins = 0
    draws = 0
    p1_kills = 0
    p2_kills = 0
    score_wins_p1 = 0
    score_wins_p2 = 0
    p1_scores = []
    p2_scores = []
    p1_lines = []
    p2_lines = []
    p1_garbage = []
    p2_garbage = []
    p1_pieces_list = []
    p2_pieces_list = []
    p1_clears_total = {1: 0, 2: 0, 3: 0, 4: 0}
    p2_clears_total = {1: 0, 2: 0, 3: 0, 4: 0}

    print(f"P1: {P1_MODEL}  vs  P2: {P2_MODEL}")
    print(f"Games: {NUM_GAMES}  Max pieces: {MAX_PIECES}\n")
    print(f"{'Game':>5}  {'Winner':<6}  {'P1 Score':>9}  {'P2 Score':>9}  {'P1 Ln':>6}  {'P2 Ln':>6}  {'P1 Garb':>7}  {'P2 Garb':>7}")
    print("-" * 68)

    for i in range(1, NUM_GAMES + 1):
        result = run_game(p1_agent, p2_agent)

        if result["winner"] == "p1":
            p1_wins += 1
            if result["cause"] == "p2_died":
                p1_kills += 1
            else:
                score_wins_p1 += 1
        elif result["winner"] == "p2":
            p2_wins += 1
            if result["cause"] == "p1_died":
                p2_kills += 1
            else:
                score_wins_p2 += 1
        else:
            draws += 1

        p1_scores.append(result["p1_score"])
        p2_scores.append(result["p2_score"])
        p1_lines.append(result["p1_lines"])
        p2_lines.append(result["p2_lines"])
        p1_garbage.append(result["p1_garbage_cleared"])
        p2_garbage.append(result["p2_garbage_cleared"])
        p1_pieces_list.append(result["p1_pieces"])
        p2_pieces_list.append(result["p2_pieces"])
        for k in p1_clears_total:
            p1_clears_total[k] += result["p1_clears"].get(k, 0)
            p2_clears_total[k] += result["p2_clears"].get(k, 0)

        print(
            f"{i:>5}  {result['winner']:<6}"
            f"  {result['p1_score']:>9}  {result['p2_score']:>9}"
            f"  {result['p1_lines']:>6}  {result['p2_lines']:>6}"
            f"  {result['p1_garbage_cleared']:>7}  {result['p2_garbage_cleared']:>7}"
        )

    p1_total_clears = sum(p1_clears_total.values())
    p2_total_clears = sum(p2_clears_total.values())
    p1_multi = p1_clears_total[2] + p1_clears_total[3] + p1_clears_total[4]
    p2_multi = p2_clears_total[2] + p2_clears_total[3] + p2_clears_total[4]
    p1_multi_pct = p1_multi / p1_total_clears * 100 if p1_total_clears else 0
    p2_multi_pct = p2_multi / p2_total_clears * 100 if p2_total_clears else 0

    print("\n" + "=" * 68)
    print(f"RESULTS — {NUM_GAMES} Games")
    print("=" * 68)
    print(f"P1 wins: {p1_wins}  P2 wins: {p2_wins}  Draws: {draws}")
    print(f"P1 win rate: {p1_wins / NUM_GAMES * 100:.1f}%  |  P2 win rate: {p2_wins / NUM_GAMES * 100:.1f}%")
    print()
    print(f"{'':16s}  {'P1':>10s}  {'P2':>10s}")
    print(f"  {'-' * 36}")
    print(f"{'Avg score':16s}  {np.mean(p1_scores):10.1f}  {np.mean(p2_scores):10.1f}")
    print(f"{'Max score':16s}  {np.max(p1_scores):10d}  {np.max(p2_scores):10d}")
    print(f"{'Avg lines':16s}  {np.mean(p1_lines):10.1f}  {np.mean(p2_lines):10.1f}")
    print(f"{'Max lines':16s}  {np.max(p1_lines):10d}  {np.max(p2_lines):10d}")
    print(f"{'Avg pieces':16s}  {np.mean(p1_pieces_list):10.1f}  {np.mean(p2_pieces_list):10.1f}")
    print(f"{'Avg garb clr':16s}  {np.mean(p1_garbage):10.1f}  {np.mean(p2_garbage):10.1f}")
    print(f"{'Lines/piece':16s}  {np.sum(p1_lines)/np.sum(p1_pieces_list):10.3f}  {np.sum(p2_lines)/np.sum(p2_pieces_list):10.3f}")
    print()
    print(f"{'Clear dist':16s}  {'P1':>10s}  {'P2':>10s}")
    print(f"  {'-' * 36}")
    print(f"{'Singles':16s}  {p1_clears_total[1]:10d}  {p2_clears_total[1]:10d}")
    print(f"{'Doubles':16s}  {p1_clears_total[2]:10d}  {p2_clears_total[2]:10d}")
    print(f"{'Triples':16s}  {p1_clears_total[3]:10d}  {p2_clears_total[3]:10d}")
    print(f"{'Tetrises':16s}  {p1_clears_total[4]:10d}  {p2_clears_total[4]:10d}")
    print(f"{'Multi-line %':16s}  {p1_multi_pct:9.1f}%  {p2_multi_pct:9.1f}%")

    pygame.quit()


if __name__ == "__main__":
    main()