"""
Tournament runner — auto-detects all models in ./models/, runs a double
round-robin, prints standings, and saves results to ./res/tournament_results.csv
"""

from itertools import permutations
from collections import defaultdict
import csv
import os
import time
import pygame

from config import P1_COMMANDS, P2_COMMANDS, BOARD_W, PADDING
from tetris import Tetris
from agents.agent import Agent


ROUNDS_PER_MATCHUP = 10
MAX_PIECES = 500


# ---------------------------------------------------------------------------
# Single game simulation
# ---------------------------------------------------------------------------


def run_game(p1_agent: Agent, p2_agent: Agent, max_pieces: int = MAX_PIECES) -> dict:
    """Simulate one headless 1v1 game. Returns per-game stats dict."""
    p1 = Tetris(x_offset=0, commands=P1_COMMANDS)
    p2 = Tetris(x_offset=BOARD_W + PADDING, commands=P2_COMMANDS)
    p1.opponent = p2
    p2.opponent = p1

    p1_piece = None
    p2_piece = None
    p1_pieces = p1_combo_count = p1_max_combo = p1_max_height = 0
    p2_pieces = p2_combo_count = p2_max_combo = p2_max_height = 0
    dt = 1000 // 60

    while True:
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
                        p1.board.copy(),
                        p1.piece,
                        p2.get_game_state()["max_height"],
                        p1.hold_piece,
                        p1.hold_used,
                        p1._get_next_piece_info(),
                    )
                    for cmd in cmds:
                        p1.execute(cmd)

                p1_max_height = max(p1_max_height, p1.get_game_state()["max_height"])
                p1_max_combo = max(p1_max_combo, p1.combo)
                if p1.combo > 0:
                    p1_combo_count += 1

                pygame.event.clear()

        if not p2.game_over:
            if p2.piece is not p2_piece:
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

                p2_max_height = max(p2_max_height, p2.get_game_state()["max_height"])
                p2_max_combo = max(p2_max_combo, p2.combo)
                if p2.combo > 0:
                    p2_combo_count += 1

                pygame.event.clear()

        p1.update(dt)
        p2.update(dt)

    p1_died = p1.game_over and not p2.game_over
    p2_died = p2.game_over and not p1.game_over

    if p1_died:
        winner, cause = "p2", "p1_died"
    elif p2_died:
        winner, cause = "p1", "p2_died"
    elif p1.score > p2.score:
        winner, cause = "p1", "score"
    elif p2.score > p1.score:
        winner, cause = "p2", "score"
    else:
        winner, cause = "draw", "draw"

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
        "p1_max_combo": p1_max_combo,
        "p2_max_combo": p2_max_combo,
        "p1_max_height": p1_max_height,
        "p2_max_height": p2_max_height,
    }


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


def run_tournament(
    players: dict[str, str], rounds_per_matchup: int = ROUNDS_PER_MATCHUP
):
    """
    Double round-robin tournament.
    players: {label: model_filename}
    Saves standings to ./res/tournament_results.csv
    """
    pygame.init()

    agents = {
        label: Agent(path, list(P1_COMMANDS.keys())) for label, path in players.items()
    }

    matchups = list(permutations(players.keys(), 2))
    total_matches = len(matchups) * rounds_per_matchup

    stats = defaultdict(
        lambda: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "kills": 0,
            "deaths": 0,
            "sum_max_height": 0,
            "score": 0,
            "lines": 0,
            "garbage_sent": 0,
            "pieces": 0,
            "combo_count": 0,
            "sum_max_combo": 0,
            "clears": {1: 0, 2: 0, 3: 0, 4: 0},
            "scores_list": [],
            "lines_list": [],
            "garbage_list": [],
        }
    )

    print(f"Models      : {len(players)}")
    print(f"Matchups    : {len(matchups)}  (double round robin)")
    print(f"Rounds each : {rounds_per_matchup}")
    print(f"Total games : {total_matches}\n")
    print(
        f"{'Game':>5}  {'P1':<16}  {'P2':<16}  {'Winner':<16}"
        f"  {'P1 Score':>10}  {'P2 Score':>10}"
        f"  {'P1 Ln':>6}  {'P2 Ln':>6}"
        f"  {'P1 Sent':>7}  {'P2 Sent':>7}  {'Time':>5}"
    )
    print("-" * 130)

    game_num = 0
    start_time = time.time()

    for home, away in matchups:
        for _ in range(rounds_per_matchup):
            game_num += 1
            result = run_game(agents[home], agents[away])

            if result["winner"] == "p1":
                winner_label = home
                stats[home]["wins"] += 1
                stats[away]["losses"] += 1
                if result["cause"] == "p2_died":
                    stats[home]["kills"] += 1
                    stats[away]["deaths"] += 1
            elif result["winner"] == "p2":
                winner_label = away
                stats[away]["wins"] += 1
                stats[home]["losses"] += 1
                if result["cause"] == "p1_died":
                    stats[away]["kills"] += 1
                    stats[home]["deaths"] += 1
            else:
                winner_label = "draw"
                stats[home]["draws"] += 1
                stats[away]["draws"] += 1

            for side, label in [("p1", home), ("p2", away)]:
                stats[label]["score"] += result[f"{side}_score"]
                stats[label]["lines"] += result[f"{side}_lines"]
                stats[label]["garbage_sent"] += result[f"{side}_garbage_sent"]
                stats[label]["pieces"] += result[f"{side}_pieces"]
                stats[label]["combo_count"] += result[f"{side}_combo_count"]
                stats[label]["sum_max_combo"] += result[f"{side}_max_combo"]
                stats[label]["sum_max_height"] += result[f"{side}_max_height"]
                stats[label]["scores_list"].append(result[f"{side}_score"])
                stats[label]["lines_list"].append(result[f"{side}_lines"])
                stats[label]["garbage_list"].append(result[f"{side}_garbage_sent"])
                for k in stats[label]["clears"]:
                    stats[label]["clears"][k] += result[f"{side}_clears"].get(k, 0)

            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            print(
                f"{game_num:>5}  {home:<16}  {away:<16}  {winner_label:<16}"
                f"  {result['p1_score']:>10}  {result['p2_score']:>10}"
                f"  {result['p1_lines']:>6}  {result['p2_lines']:>6}"
                f"  {result['p1_garbage_sent']:>7}  {result['p2_garbage_sent']:>7}"
                f"  {m}:{s:02d}"
            )

    games_per_player = (len(players) - 1) * 2 * rounds_per_matchup

    standings = sorted(
        stats.items(),
        key=lambda x: (x[1]["wins"], x[1]["draws"], x[1]["score"]),
        reverse=True,
    )

    # --- print standings ---
    print("\n" + "=" * 112)
    print("STANDINGS")
    print(
        f"  {'Player':<16}  {'W':>4}  {'L':>4}  {'D':>4}  {'Kills':>5}"
        f"  {'Win%':>6}  {'Avg Score':>10}  {'Avg Lines':>10}  {'Avg Sent':>9}"
    )
    print("  " + "-" * 82)
    for label, s in standings:
        avg_score = s["score"] / games_per_player if games_per_player else 0
        avg_lines = s["lines"] / games_per_player if games_per_player else 0
        avg_sent = s["garbage_sent"] / games_per_player if games_per_player else 0
        win_pct = s["wins"] / games_per_player * 100 if games_per_player else 0
        print(
            f"  {label:<16}  {s['wins']:>4}  {s['losses']:>4}  {s['draws']:>4}  {s['kills']:>5}"
            f"  {win_pct:>5.1f}%  {avg_score:>10.1f}  {avg_lines:>10.1f}  {avg_sent:>9.1f}"
        )

    print("\n" + "=" * 112)
    print("DETAILED STATS")
    print(
        f"  {'Player':<16}  {'Ln/Pc':>6}  {'Singles':>8}  {'Doubles':>8}"
        f"  {'Triples':>8}  {'Tetrises':>8}  {'Combos':>7}  {'Cmb/Game':>9}"
    )
    print("  " + "-" * 82)
    for label, s in standings:
        lpp = s["lines"] / s["pieces"] if s["pieces"] else 0
        cpg = s["combo_count"] / games_per_player if games_per_player else 0
        print(
            f"  {label:<16}  {lpp:>6.3f}  {s['clears'][1]:>8}  {s['clears'][2]:>8}"
            f"  {s['clears'][3]:>8}  {s['clears'][4]:>8}"
            f"  {s['combo_count']:>7}  {cpg:>9.1f}"
        )

    # --- save CSV ---
    os.makedirs("./res", exist_ok=True)
    csv_path = "./res/tournament_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "player",
                "model",
                "wins",
                "losses",
                "draws",
                "kills",
                "deaths",
                "win_pct",
                "avg_score",
                "avg_lines",
                "avg_garbage_sent",
                "avg_max_height",
                "lines_per_piece",
                "singles",
                "doubles",
                "triples",
                "tetrises",
                "total_combos",
                "combos_per_game",
                "avg_max_combo",
            ]
        )
        for label, s in standings:
            avg_score = s["score"] / games_per_player if games_per_player else 0
            avg_lines = s["lines"] / games_per_player if games_per_player else 0
            avg_sent = s["garbage_sent"] / games_per_player if games_per_player else 0
            avg_max_height = (
                s["sum_max_height"] / games_per_player if games_per_player else 0
            )
            win_pct = s["wins"] / games_per_player * 100 if games_per_player else 0
            lpp = s["lines"] / s["pieces"] if s["pieces"] else 0
            cpg = s["combo_count"] / games_per_player if games_per_player else 0
            avg_max_combo = (
                s["sum_max_combo"] / games_per_player if games_per_player else 0
            )
            writer.writerow(
                [
                    label,
                    players[label],
                    s["wins"],
                    s["losses"],
                    s["draws"],
                    s["kills"],
                    s["deaths"],
                    round(win_pct, 1),
                    round(avg_score, 1),
                    round(avg_lines, 1),
                    round(avg_sent, 1),
                    round(avg_max_height, 1),
                    round(lpp, 3),
                    s["clears"][1],
                    s["clears"][2],
                    s["clears"][3],
                    s["clears"][4],
                    s["combo_count"],
                    round(cpg, 1),
                    round(avg_max_combo, 1),
                ]
            )

    total_time = time.time() - start_time
    m, s = divmod(int(total_time), 60)
    print(f"\nCompleted in {m}:{s:02d}")
    print(f"Results saved → {csv_path}")

    pygame.quit()
    return dict(stats)


# ---------------------------------------------------------------------------
# Entry point — auto-detect all models
# ---------------------------------------------------------------------------

_ABBREV = {
    "defensive": "def",
    "offensive": "off",
    "neutral": "neu",
    "heuristic": "heu",
    "hybrid": "hyb",
    "random": "rnd",
}


def _label(filename: str) -> str:
    """Convert 'tetris_dqn_defensive_vs_heuristic_v1.keras' → 'def_vs_heu_v1'"""
    name = filename.replace(".keras", "").replace("tetris_dqn_", "")
    parts = name.split("_")
    return "_".join(_ABBREV.get(p, p) for p in parts)


if __name__ == "__main__":
    models_dir = "./models"
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".keras"):
                rel = os.path.relpath(os.path.join(root, f), models_dir)
                model_files.append(rel)
    model_files = sorted(model_files)
    if not model_files:
        print("No .keras models found in ./models/")
    else:
        players = {_label(os.path.basename(f)): f for f in model_files}
        print(f"Found {len(players)} models:")
        for label, path in players.items():
            print(f"  {label:<16} → {path}")
        print()
        run_tournament(players, rounds_per_matchup=ROUNDS_PER_MATCHUP)
