"""Train agent, outputs a saved model file"""

import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
from collections import deque
from tensorflow import keras
from config import ROWS, COLS, TETROMINOS
from tetris import Tetris, Piece
from models.pathfinder import Pathfinder

AGENT_COMMANDS = {
    "left": (0, "a"),
    "right": (0, "d"),
    "down": (0, "s"),
    "rotate": (0, "w"),
    "drop": (0, "p"),
}
OPP_COMMANDS = {
    "left": (0, "l"),
    "right": (0, "r"),
    "down": (0, "n"),
    "rotate": (0, "u"),
    "drop": (0, "0"),
}

NUM_PIECES = len(TETROMINOS)  # 7
STATE_SIZE = 4 + NUM_PIECES + 1  # board features + piece one-hot + opp height
MEM_SIZE = 20000
BATCH_SIZE = 64
DISCOUNT = 0.95
EPOCHS = 1
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_STOP_EP = 5000
REPLAY_START = 1000
TRAIN_EPISODES = 8000
TARGET_UPDATE = 200
SAVE_PATH = "tetris_dqn.keras"


def make_state(
    board: np.ndarray, lines_cleared: int, piece: Piece, opp_agg: int
) -> np.ndarray:
    """Build a state vector from post-action board features.

    Used identically in best_action (to score candidates) and in replay
    (as the training input), so the model trains on the same distribution
    it infers on.
    """
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
    return np.concatenate([own, piece_oh, [opp_agg]])


def copy_piece(piece: Piece) -> Piece:
    """Snapshot a Piece so later mutations don't corrupt replay memory."""
    p = Piece(piece.shape, piece.color_id)
    p.row = piece.row
    p.col = piece.col
    p.rotation_id = piece.rotation_id
    return p


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

    def best_action(self, actions, piece, opp_agg):
        if random.random() < self.epsilon:
            return random.choice(actions)
        states = np.array(
            [
                make_state(a["board_result"], a["lines_cleared"], piece, opp_agg)
                for a in actions
            ]
        )
        qs = self.model(states, training=False).numpy().flatten()
        return actions[int(np.argmax(qs))]

    # memory tuple: (action_state, reward, done, next_board, next_piece, opp_agg)
    def remember(self, action_state, reward, done, next_board, next_piece, opp_agg):
        self.memory.append(
            (action_state, reward, done, next_board, next_piece, opp_agg)
        )

    def train(self, pf: Pathfinder):
        if len(self.memory) < REPLAY_START:
            return
        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        x, y = [], []
        for action_state, reward, done, next_board, next_piece, opp_agg in batch:
            if done:
                target = reward
            else:
                next_actions = pf.get_actions(next_board, next_piece)
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

    def save(self):
        self.model.save(SAVE_PATH)
        print(f"Saved → {SAVE_PATH}")


def play_episode(
    p1: Tetris, p2: Tetris, agent: DQNAgent, pf: Pathfinder, max_pieces: int = 1000
):
    total_reward = 0
    pieces = 0
    prev_height = p1.get_game_state()["aggregate_height"]
    while not p1.game_over and not p2.game_over and pieces < max_pieces:
        pieces += 1
        actions = pf.get_actions(p1.board.copy(), p1.piece)
        if not actions:
            break

        opp_agg = p2.get_game_state()["aggregate_height"]
        total_before = p1.normal_lines_cleared
        garbage_before = p1.garbage_lines_cleared
        chosen = agent.best_action(actions, p1.piece, opp_agg)

        for cmd in chosen["sequence"]:
            p1.execute(cmd)
        pygame.event.clear()

        """Heuristic for Opponent"""

        def heuristic_score(action):
            b = action["board_result"]
            heights = [
                next((ROWS - r for r in range(ROWS) if b[r][c]), 0) for c in range(COLS)
            ]
            agg = sum(heights)
            holes = sum(
                1
                for c in range(COLS)
                for r in range(ROWS - heights[c], ROWS)
                if not b[r][c]
            )
            bump = sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1))
            return action["lines_cleared"] * 100 - agg * 0.2 - holes * 0.5 - bump * 0.1

        # random opponent step
        opp_acts = pf.get_actions(p2.board.copy(), p2.piece)
        if opp_acts:
            for cmd in max(opp_acts, key=heuristic_score)["sequence"]:
                p2.execute(cmd)
            pygame.event.clear()

        normal_cleared = p1.normal_lines_cleared - total_before
        garbage_cleared = p1.garbage_lines_cleared - garbage_before
        opp_agg_after = p2.get_game_state()["aggregate_height"]

        # build action_state from the simulated post-placement board
        # this matches exactly what best_action scored during selection
        action_state = make_state(
            chosen["board_result"], normal_cleared, p1.piece, opp_agg_after
        )

        gs = p1.get_game_state()
        height_delta = gs["aggregate_height"] - prev_height
        line_rewards = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        reward = (
            line_rewards.get(normal_cleared, 800)
            + garbage_cleared * 50
            + 1
            - gs["holes"] * 0.5
            - gs["bumpiness"] * 0.1
            - max(height_delta, 0) * 0.5
            + opp_agg_after * 1.0
        )
        if p1.game_over:
            reward = -500
        elif p2.game_over:
            reward += 20

        prev_height = gs["aggregate_height"]

        done = p1.game_over or p2.game_over
        agent.remember(
            action_state,
            reward,
            done,
            p1.board.copy(),
            copy_piece(p1.piece),
            opp_agg_after,
        )
        total_reward += reward

    return total_reward


def train():
    if not pygame.get_init():
        pygame.init()
        pygame.display.set_mode((1, 1))

    pf = Pathfinder()
    agent = DQNAgent()
    best_score = -np.inf

    rewards = []
    epsilons = []
    high_rewards = []
    for ep in range(1, TRAIN_EPISODES + 1):
        p1 = Tetris(x_offset=0, commands=AGENT_COMMANDS)
        p2 = Tetris(x_offset=0, commands=OPP_COMMANDS)
        p1.opponent = p2
        p2.opponent = p1
        garbage = random.randint(0, 14)
        p1.respawn_garbage_lines(garbage)

        total_reward = play_episode(p1, p2, agent, pf)

        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        if garbage >= 12:
            high_rewards.append(total_reward)

        agent.train(pf)
        agent.decay_epsilon()

        if total_reward > best_score:
            best_score = total_reward
            agent.save()

        if ep % 50 == 0:
            high_avg = np.mean(high_rewards[-20:]) if high_rewards else 0
            print(
                f"ep={ep:4d}  reward={total_reward:7.1f}  eps={agent.epsilon:.3f}  "
                f"best={best_score:.1f}  high_avg={high_avg:.1f}"
            )
        if ep % 200 == 0:
            draw_figure(rewards, epsilons)

    print("Training complete.")


def draw_figure(rewards, epsilons):
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
    fig.savefig("rewards.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(episodes, epsilons)
    ax.set(xlabel="Episode", ylabel="Epsilon", title="DQN epsilons")
    fig.savefig("epsilons.png")
    plt.close(fig)


if __name__ == "__main__":
    train()
