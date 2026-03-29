"""Train agent, outputs a saved model file"""

import numpy as np
import pygame
import random
from collections import deque
from tensorflow import keras
from config import ROWS, COLS, TETROMINOS, SCORE_TABLE
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
TRAIN_EPISODES = 5000
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
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-3))
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
        qs = self.model.predict(states, verbose=0).flatten()
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
                        self.target_model.predict(next_states, verbose=0).flatten()
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
    p1: Tetris, p2: Tetris, agent: DQNAgent, pf: Pathfinder, max_pieces: int = 500
):
    total_reward = 0
    pieces = 0
    while not p1.game_over and not p2.game_over and pieces < max_pieces:
        pieces += 1
        actions = pf.get_actions(p1.board.copy(), p1.piece)
        if not actions:
            break

        opp_agg = p2.get_game_state()["aggregate_height"]
        score_before = p1.score
        chosen = agent.best_action(actions, p1.piece, opp_agg)

        for cmd in chosen["sequence"]:
            p1.execute(cmd)
        pygame.event.clear()

        # random opponent step
        opp_acts = pf.get_actions(p2.board.copy(), p2.piece)
        if opp_acts:
            for cmd in random.choice(opp_acts)["sequence"]:
                p2.execute(cmd)
            pygame.event.clear()

        lines_cleared = p1.score - score_before
        opp_agg_after = p2.get_game_state()["aggregate_height"]

        # build action_state from the simulated post-placement board
        # this matches exactly what best_action scored during selection
        action_state = make_state(
            chosen["board_result"], lines_cleared, p1.piece, opp_agg_after
        )

        gs = p1.get_game_state()

        reward = (
            SCORE_TABLE.get(lines_cleared, 800)
            - gs["holes"] * 0.5
            - gs["bumpiness"] * 0.1
            - gs["aggregate_height"] * 0.2
            + opp_agg_after * 1.0
        )
        if p1.game_over:
            reward = -500
        elif p2.game_over:
            reward += 20

        done = p1.game_over or p2.game_over
        agent.remember(
            action_state,
            reward,
            done,
            p1.board.copy(),
            p1.piece.copy(),
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

    for ep in range(1, TRAIN_EPISODES + 1):
        p1 = Tetris(x_offset=0, commands=AGENT_COMMANDS)
        p2 = Tetris(x_offset=0, commands=OPP_COMMANDS)
        p1.opponent = p2
        p2.opponent = p1

        total_reward = play_episode(p1, p2, agent, pf)
        agent.train(pf)
        agent.decay_epsilon()

        if total_reward > best_score:
            best_score = total_reward
            agent.save()

        if ep % 50 == 0:
            print(
                f"ep={ep:4d}  reward={total_reward:7.1f}  eps={agent.epsilon:.3f}  best={best_score:.1f}"
            )

    print("Training complete.")


if __name__ == "__main__":
    train()
