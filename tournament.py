from itertools import permutations
from collections import defaultdict
import numpy as np
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
    p1_combo_count = 0
    p2_combo_count = 0
    dt = 1000 // 60

    for _ in range(max_ticks):
        pygame.event.clear()

        if p1.game_over or p2.game_over:
            break
        if p1_pieces >= max_pieces and p2_pieces >= max_pieces:
            break

        if not p1.game_over:
            if p1.piece is not p1_piece:
                p1_prev_combo = p1.combo
                p1_piece = p1.piece
                p1_pieces += 1
                if p1_pieces <= max_pieces:
                    cmds = p1_agent.get_command_sequence(
                        p1.board.copy(),
                        p1.piece,
                        p2.get_game_state()["max_height"],
                        p1.hold_piece,
                        p1.hold_used,
                        p1._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p1.execute(cmd)
                if p1.combo > p1_prev_combo:
                    p1_combo_count += 1
                pygame.event.clear()

        if not p2.game_over:
            if p2.piece is not p2_piece:
                p2_prev_combo = p2.combo
                p2_piece = p2.piece
                p2_pieces += 1
                if p2_pieces <= max_pieces:
                    cmds = p2_agent.get_command_sequence(
                        p2.board.copy(),
                        p2.piece,
                        p1.get_game_state()["max_height"],
                        p2.hold_piece,
                        p2.hold_used,
                        p2._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p2.execute(cmd)
                if p2.combo > p2_prev_combo:
                    p2_combo_count += 1
                pygame.event.clear()

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
        "p1_lines": p1.normal_lines_cleared + p1.garbage_lines_cleared,
        "p2_lines": p2.normal_lines_cleared + p2.garbage_lines_cleared,
        "p1_garbage_sent": p1.total_garbage_sent,
        "p2_garbage_sent": p2.total_garbage_sent,
        "p1_pieces": p1_pieces,
        "p2_pieces": p2_pieces,
        "p1_clears": dict(p1.clear_distribution),
        "p2_clears": dict(p2.clear_distribution),
        "p1_height": p1.get_game_state()["max_height"],
        "p2_height": p2.get_game_state()["max_height"],
        "p1_combo_count": p1_combo_count,
        "p2_combo_count": p2_combo_count,
    }


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

    stats = defaultdict(lambda: {
        "wins": 0, "losses": 0, "draws": 0, "kills": 0,
        "score": 0, "lines": 0, "garbage_sent": 0, "pieces": 0,
        "clears": {1: 0, 2: 0, 3: 0, 4: 0},
        "combo_count": 0, "scores_list": [], "lines_list": [],
        "garbage_list": [],
    })

    print(f"Players     : {', '.join(f'{k}={v}' for k, v in players.items())}")
    print(f"Matchups    : {len(matchups)}  (double round robin)")
    print(f"Rounds each : {rounds_per_matchup}")
    print(f"Total games : {total_matches}\n")
    print(
        f"{'Game':>5}  {'P1':<10}  {'P2':<10}  {'Winner':<10}"
        f"  {'P1 Score':>10}  {'P2 Score':>10}"
        f"  {'P1 Ln':>6}  {'P2 Ln':>6}"
        f"  {'P1 Sent':>7}  {'P2 Sent':>7}"
    )
    print("-" * 90)

    game_num = 0
    for home, away in matchups:
        for _ in range(rounds_per_matchup):
            game_num += 1
            result = run_headless_round(agents[home], agents[away])

            if result["winner"] == "p1":
                winner_label = home
                stats[home]["wins"] += 1
                stats[away]["losses"] += 1
                if result["cause"] == "p2_died":
                    stats[home]["kills"] += 1
            elif result["winner"] == "p2":
                winner_label = away
                stats[away]["wins"] += 1
                stats[home]["losses"] += 1
                if result["cause"] == "p1_died":
                    stats[away]["kills"] += 1
            else:
                winner_label = "draw"
                stats[home]["draws"] += 1
                stats[away]["draws"] += 1

            # accumulate stats for home (as P1)
            stats[home]["score"] += result["p1_score"]
            stats[home]["lines"] += result["p1_lines"]
            stats[home]["garbage_sent"] += result["p1_garbage_sent"]
            stats[home]["pieces"] += result["p1_pieces"]
            stats[home]["combo_count"] += result["p1_combo_count"]
            stats[home]["scores_list"].append(result["p1_score"])
            stats[home]["lines_list"].append(result["p1_lines"])
            stats[home]["garbage_list"].append(result["p1_garbage_sent"])
            for k in stats[home]["clears"]:
                stats[home]["clears"][k] += result["p1_clears"].get(k, 0)

            # accumulate stats for away (as P2)
            stats[away]["score"] += result["p2_score"]
            stats[away]["lines"] += result["p2_lines"]
            stats[away]["garbage_sent"] += result["p2_garbage_sent"]
            stats[away]["pieces"] += result["p2_pieces"]
            stats[away]["combo_count"] += result["p2_combo_count"]
            stats[away]["scores_list"].append(result["p2_score"])
            stats[away]["lines_list"].append(result["p2_lines"])
            stats[away]["garbage_list"].append(result["p2_garbage_sent"])
            for k in stats[away]["clears"]:
                stats[away]["clears"][k] += result["p2_clears"].get(k, 0)

            print(
                f"{game_num:>5}  {home:<10}  {away:<10}  {winner_label:<10}"
                f"  {result['p1_score']:>10}  {result['p2_score']:>10}"
                f"  {result['p1_lines']:>6}  {result['p2_lines']:>6}"
                f"  {result['p1_garbage_sent']:>7}  {result['p2_garbage_sent']:>7}"
            )

    standings = sorted(
        stats.items(),
        key=lambda x: (x[1]["wins"], x[1]["draws"], x[1]["score"]),
        reverse=True,
    )

    games_per_player = (len(players) - 1) * 2 * rounds_per_matchup

    print("\n" + "=" * 90)
    print("STANDINGS")
    print(
        f"  {'Player':<10}  {'W':>4}  {'L':>4}  {'D':>4}  {'Kills':>5}"
        f"  {'Win %':>7}  {'Avg Score':>10}  {'Avg Lines':>10}  {'Avg Sent':>9}"
    )
    print("  " + "-" * 72)
    for label, s in standings:
        avg_score = s["score"] / games_per_player if games_per_player else 0
        avg_lines = s["lines"] / games_per_player if games_per_player else 0
        avg_sent = s["garbage_sent"] / games_per_player if games_per_player else 0
        win_pct = s["wins"] / games_per_player * 100 if games_per_player else 0
        print(
            f"  {label:<10}  {s['wins']:>4}  {s['losses']:>4}  {s['draws']:>4}  {s['kills']:>5}"
            f"  {win_pct:>6.1f}%  {avg_score:>10.1f}  {avg_lines:>10.1f}  {avg_sent:>9.1f}"
        )

    print("\n" + "=" * 90)
    print("DETAILED STATS")
    print(
        f"  {'Player':<10}  {'Lines/Pc':>9}  {'Singles':>8}  {'Doubles':>8}"
        f"  {'Triples':>8}  {'Tetrises':>8}  {'Combos':>7}  {'Comb/Game':>9}"
    )
    print("  " + "-" * 72)
    for label, s in standings:
        lpp = s["lines"] / s["pieces"] if s["pieces"] else 0
        cpg = s["combo_count"] / games_per_player if games_per_player else 0
        print(
            f"  {label:<10}  {lpp:>9.3f}  {s['clears'][1]:>8}  {s['clears'][2]:>8}"
            f"  {s['clears'][3]:>8}  {s['clears'][4]:>8}"
            f"  {s['combo_count']:>7}  {cpg:>9.1f}"
        )

    pygame.quit()
    return dict(stats)


if __name__ == "__main__":
    players = {
        "nvhe_v1": "tetris_dqn_neutral_vs_heuristic_v1.keras",
        "nvhy_v1": "tetris_dqn_neutral_vs_hybrid_v1.keras",
        "ovhe_v1": "tetris_dqn_offensive_vs_heuristic_v1.keras",
        "ovhe_v2": "tetris_dqn_offensive_vs_heuristic_v2.keras",
    }
    run_double_round_robin(players, rounds_per_matchup=10)