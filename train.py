"""Train agent, outputs a saved model file"""

import numpy as np
import pygame
import random
import time
import matplotlib.pyplot as plt
from collections import deque
from tensorflow import keras
from config import ROWS, COLS, TETROMINOS
from tetris import Tetris, Piece
from agents.pathfinder import Pathfinder
from agents.strategy import Strategy

AGENT_COMMANDS = {
    "left": (0, "a"),
    "right": (0, "d"),
    "down": (0, "s"),
    "rotate": (0, "w"),
    "hold": (0, "q"),
    "drop": (0, "p"),
}
OPP_COMMANDS = {
    "left": (0, "l"),
    "right": (0, "r"),
    "down": (0, "n"),
    "rotate": (0, "u"),
    "hold": (0, "/"),
    "drop": (0, "0"),
}

NUM_PIECES = len(TETROMINOS)  # 7x
# board features(4) + current piece one-hot(7) + hold piece one-hot(7) + opp height(1) + hold_available(1)
STATE_SIZE = 4 + NUM_PIECES + NUM_PIECES + 1 + 1
MEM_SIZE = 50000
BATCH_SIZE = 128
MAX_PIECES = 250
DISCOUNT = 0.95
EPOCHS = 1
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_STOP_EP = 1500
REPLAY_START = 1000
TRAIN_EPISODES = 2000
TARGET_UPDATE = 200
STRATEGIES = ["neutral", "offensive", "defensive"]


def make_state(
    board: np.ndarray,
    lines_cleared: int,
    piece: Piece,
    opp_agg: int,
    hold_piece=None,
    hold_used=False,
) -> np.ndarray:
    """Build a state vector from post-action board features."""
    heights = np.zeros(COLS, dtype=int)
    for col in range(COLS):
        for row in range(ROWS):
            if board[row][col]:
                heights[col] = ROWS - row
                break
    agg = int(heights.max())
    bump = int(sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1)))
    holes = sum(
        1
        for c in range(COLS)
        if heights[c]
        for r in range(ROWS - heights[c], ROWS)
        if not board[r][c]
    )
    own = np.array([agg, holes, bump, lines_cleared], dtype=np.float32)

    piece_oh = np.zeros(NUM_PIECES, dtype=np.float32)
    piece_oh[piece.color_id - 1] = 1.0

    hold_oh = np.zeros(NUM_PIECES, dtype=np.float32)
    if hold_piece is not None:
        _, hold_color_id = hold_piece
        hold_oh[hold_color_id - 1] = 1.0

    hold_avail = np.array([0.0 if hold_used else 1.0], dtype=np.float32)

    return np.concatenate([own, piece_oh, hold_oh, [opp_agg], hold_avail])


def build_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(STATE_SIZE,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(loss="huber", optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    return model


class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=MEM_SIZE)
        self.epsilon = EPSILON_START
        self._decay = (EPSILON_START - EPSILON_MIN) / EPSILON_STOP_EP
        self.update_counter = 0

    def best_action(self, actions, piece, opp_agg, hold_piece=None, hold_used=False):
        if random.random() < self.epsilon:
            return random.choice(actions)
        states = np.array(
            [
                make_state(
                    a["board_result"],
                    a["lines_cleared"],
                    piece,
                    opp_agg,
                    hold_piece,
                    hold_used,
                )
                for a in actions
            ]
        )
        qs = self.model(states, training=False).numpy().flatten()
        return actions[int(np.argmax(qs))]

    def remember(
        self,
        action_state,
        reward,
        done,
        next_board,
        next_piece,
        opp_agg,
        hold_piece,
        hold_used,
        next_piece_info,
    ):
        self.memory.append(
            (
                action_state,
                reward,
                done,
                next_board,
                next_piece,
                opp_agg,
                hold_piece,
                hold_used,
                next_piece_info,
            )
        )

    def train(self, pf: Pathfinder):
        if len(self.memory) < REPLAY_START:
            return
        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        x, y = [], []
        for (
            action_state,
            reward,
            done,
            next_board,
            next_piece,
            opp_agg,
            hold_piece,
            hold_used,
            next_piece_info,
        ) in batch:
            if done:
                target = reward
            else:
                next_actions = pf.get_actions(
                    next_board, next_piece, hold_piece, hold_used, next_piece_info
                )
                if not next_actions:
                    target = reward
                else:
                    next_states = np.array(
                        [
                            make_state(
                                a["board_result"],
                                a["lines_cleared"],
                                next_piece,
                                opp_agg,
                                hold_piece,
                                hold_used,
                            )
                            for a in next_actions
                        ]
                    )
                    target = reward + DISCOUNT * np.max(
                        self.target_model(next_states, training=False).numpy().flatten()
                    )
            x.append(action_state)
            y.append(target)
        self.model.fit(
            np.array(x), np.array(y), batch_size=len(x), epochs=EPOCHS, verbose=0
        )
        self.update_counter += 1
        if self.update_counter % TARGET_UPDATE == 0:
            self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon - self._decay)

    def save(self, path):
        self.model.save(path)
        print(f"Saved → {path}")


def _placed_piece(action, piece, hold_piece, next_piece_info):
    """Return the Piece that was actually placed for this action."""
    if action["sequence"] and action["sequence"][0] == "hold":
        if hold_piece is not None:
            return Piece(hold_piece[0], hold_piece[1])
        elif next_piece_info is not None:
            return Piece(next_piece_info[0], next_piece_info[1])
    return piece


def play_episode(
    p1: Tetris,
    p2: Tetris,
    agent: DQNAgent,
    pf: Pathfinder,
    strategy: Strategy,
    strategy_name: str,
    max_pieces: int = 1000,
):
    total_reward = 0
    pieces = 0
    prev_height = p1.get_game_state()["max_height"]
    while not p1.game_over and not p2.game_over and pieces < max_pieces:
        pieces += 1
        snap_next = p1._get_next_piece_info()
        actions = pf.get_actions(
            p1.board.copy(), p1.piece, p1.hold_piece, p1.hold_used, snap_next
        )
        if not actions:
            break

        opp_agg = p2.get_game_state()["max_height"]
        total_before = p1.normal_lines_cleared
        garbage_before = p1.garbage_lines_cleared

        snap_piece = p1.piece
        snap_hold = p1.hold_piece
        snap_hold_used = p1.hold_used

        chosen = agent.best_action(
            actions, snap_piece, opp_agg, snap_hold, snap_hold_used
        )

        for cmd in chosen["sequence"]:
            p1.execute(cmd)
        pygame.event.clear()

        opp_next_info = p2._get_next_piece_info()
        opp_acts = pf.get_actions(
            p2.board.copy(), p2.piece, p2.hold_piece, p2.hold_used, opp_next_info
        )
        if opp_acts:
            for cmd in max(opp_acts, key=strategy.get_heuristic)["sequence"]:
                p2.execute(cmd)
            pygame.event.clear()

        normal_cleared = p1.normal_lines_cleared - total_before
        garbage_cleared = p1.garbage_lines_cleared - garbage_before
        total_cleared = normal_cleared + garbage_cleared

        opp_agg_after = p2.get_game_state()["max_height"]

        gs = p1.get_game_state()
        height_delta = gs["max_height"] - prev_height

        reward = strategy.get_reward(
            strategy_name,
            total_cleared,
            gs["holes"],
            gs["bumpiness"],
            gs["max_height"],
            height_delta,
        )

        if p1.game_over:
            reward = strategy.penalties[strategy_name]["death"]
        elif p2.game_over:
            reward += strategy.penalties[strategy_name]["win"]

        prev_height = gs["max_height"]

        done = p1.game_over or p2.game_over

        placed = _placed_piece(chosen, snap_piece, snap_hold, snap_next)
        action_state = make_state(
            chosen["board_result"],
            total_cleared,
            placed,
            opp_agg_after,
            p1.hold_piece,
            p1.hold_used,
        )
        agent.remember(
            action_state,
            reward,
            done,
            p1.board.copy(),
            p1.piece.copy(),
            opp_agg_after,
            p1.hold_piece,
            p1.hold_used,
            p1._get_next_piece_info(),
        )
        total_reward += reward

    return total_reward


def train():
    print("Select strategy:")
    for i, name in enumerate(STRATEGIES, 1):
        print(f"  {i}. {name}")
    choice = input("Enter 1/2/3: ").strip()
    while choice not in ("1", "2", "3"):
        choice = input("Invalid. Enter 1/2/3: ").strip()
    strategy_name = STRATEGIES[int(choice) - 1]
    save_path = f"./models/tetris_dqn_{strategy_name}.keras"
    print(f"Training: {strategy_name} → {save_path}\n")

    if not pygame.get_init():
        pygame.init()
        pygame.display.set_mode((1, 1))

    pf = Pathfinder()
    agent = DQNAgent()
    strategy = Strategy()

    best_score = -np.inf
    start_time = time.time()

    rewards = []
    for ep in range(1, TRAIN_EPISODES + 1):
        p1 = Tetris(x_offset=0, commands=AGENT_COMMANDS)
        p2 = Tetris(x_offset=0, commands=OPP_COMMANDS)
        p1.opponent = p2
        p2.opponent = p1
        garbage = random.randint(0, 14)
        p1.respawn_garbage_lines(garbage)

        total_reward = play_episode(p1, p2, agent, pf, strategy, strategy_name, max_pieces=MAX_PIECES)

        rewards.append(total_reward)

        agent.train(pf)
        agent.decay_epsilon()

        if ep % 50 == 0:
            avg_50 = np.mean(rewards[-50:])
            if avg_50 > best_score:
                best_score = avg_50
                agent.save(save_path)
            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            print(
                f"ep={ep:4d}  avg50={avg_50:7.1f}  eps={agent.epsilon:.3f}  "
                f"best={best_score:.1f}  time={h}:{m:02d}:{s:02d}"
            )
        if ep % 200 == 0:
            draw_figure(rewards, strategy_name)

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"Training complete. Total time: {h}:{m:02d}:{s:02d}")


def draw_figure(rewards, strategy_name):
    window = 50
    episodes = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, alpha=0.3, label="Raw")

    if len(rewards) >= window:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            rolling_avg,
            label=f"Rolling average (n={window})",
        )

    ax.set(xlabel="Episode", ylabel="Reward", title="DQN rewards")
    ax.legend()
    fig.savefig(f"./res/rewards_{strategy_name}.png")
    plt.close(fig)


if __name__ == "__main__":
    train()