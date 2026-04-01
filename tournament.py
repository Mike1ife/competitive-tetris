from itertools import permutations
from collections import defaultdict
import pygame
from config import P1_COMMANDS, P2_COMMANDS, BOARD_W, PADDING
from tetris import Tetris
from agents.agent import Agent


def run_headless_round(
    p1_agent: Agent, p2_agent: Agent, max_ticks: int = 20000, max_pieces: int = 500
) -> dict:
    """Simulate one full game between two agents without any rendering."""
    p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
    p2 = Tetris(x_offset=BOARD_W + PADDING, commands=P2_COMMANDS)
    p1.opponent = p2
    p2.opponent = p1

    p1_piece = None
    p2_piece = None
    p1_pieces = 0
    p2_pieces = 0
    dt = 1000 // 60

    for _ in range(max_ticks):
        pygame.event.clear()

        if p1.game_over or p2.game_over:
            break
        if p1_pieces >= max_pieces and p2_pieces >= max_pieces:
            break

        if not p1.game_over:
            if p1.piece is not p1_piece:
                p1_piece = p1.piece
                p1_pieces += 1
                if p1_pieces <= max_pieces:
                    cmds = p1_agent.get_command_sequence(
                        p1.board.copy(), p1.piece,
                        p2.get_game_state()["max_height"],
                        p1.hold_piece, p1.hold_used,
                        p1._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p1.execute(cmd)
                pygame.event.clear()

        if not p2.game_over:
            if p2.piece is not p2_piece:
                p2_piece = p2.piece
                p2_pieces += 1
                if p2_pieces <= max_pieces:
                    cmds = p2_agent.get_command_sequence(
                        p2.board.copy(), p2.piece,
                        p1.get_game_state()["max_height"],
                        p2.hold_piece, p2.hold_used,
                        p2._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p2.execute(cmd)
                pygame.event.clear()

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

    return {"winner": winner, "p1_score": p1.score, "p2_score": p2.score}


def run_double_round_robin(players: dict[str, str], rounds_per_matchup: int = 1):
    """
    Run a double round robin tournament.

    Each ordered pair (A, B) plays `rounds_per_matchup` games, so every
    player faces every other player as both P1 and P2.
    """
    pygame.init()

    agents = {
        label: Agent(path, list(P1_COMMANDS.keys())) for label, path in players.items()
    }

    matchups = list(permutations(players.keys(), 2))
    total_matches = len(matchups) * rounds_per_matchup

    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "score": 0})

    print(f"Players     : {', '.join(f'{k}={v}' for k, v in players.items())}")
    print(f"Matchups    : {len(matchups)}  (double round robin)")
    print(f"Rounds each : {rounds_per_matchup}")
    print(f"Total games : {total_matches}\n")
    print(
        f"{'Game':>5}  {'P1':<6}  {'P2':<6}  {'Winner':<8}  {'P1 Score':>10}  {'P2 Score':>10}"
    )
    print("-" * 56)

    game_num = 0
    for home, away in matchups:
        for _ in range(rounds_per_matchup):
            game_num += 1
            result = run_headless_round(agents[home], agents[away])

            if result["winner"] == "p1":
                winner_label = home
                stats[home]["wins"] += 1
                stats[away]["losses"] += 1
            elif result["winner"] == "p2":
                winner_label = away
                stats[away]["wins"] += 1
                stats[home]["losses"] += 1
            else:
                winner_label = "draw"
                stats[home]["draws"] += 1
                stats[away]["draws"] += 1

            stats[home]["score"] += result["p1_score"]
            stats[away]["score"] += result["p2_score"]

            print(
                f"{game_num:>5}  {home:<6}  {away:<6}  {winner_label:<8}"
                f"  {result['p1_score']:>10}  {result['p2_score']:>10}"
            )

    standings = sorted(
        stats.items(),
        key=lambda x: (x[1]["wins"], x[1]["draws"], x[1]["score"]),
        reverse=True,
    )

    games_per_player = (len(players) - 1) * 2 * rounds_per_matchup

    print("\n" + "=" * 56)
    print("STANDINGS")
    print(f"  {'Player':<8}  {'W':>4}  {'L':>4}  {'D':>4}  {'Avg Score':>10}")
    print("  " + "-" * 38)
    for label, s in standings:
        avg = s["score"] / games_per_player if games_per_player else 0
        print(
            f"  {label:<8}  {s['wins']:>4}  {s['losses']:>4}  {s['draws']:>4}  {avg:>10.1f}"
        )

    pygame.quit()
    return dict(stats)


if __name__ == "__main__":
    players = {
        "p1": "tetris_dqn_neutral.keras",
        "p2": "tetris_dqn_offensive_v1.keras",
        "p3": "tetris_dqn_defensive_v3.keras",
        "p4": "tetris_dqn_defensive_v2.keras",
    }
    run_double_round_robin(players, rounds_per_matchup=5)