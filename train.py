"""Train agent, outputs a saved model file"""

import numpy as np
import pygame
import random
from collections import deque
from tensorflow import keras
from config import ROWS, COLS
from tetris import Tetris
from models.pathfinder import Pathfinder

AGENT_COMMANDS = {
    "left": (0, "a"), "right": (0, "d"), "down": (0, "s"),
    "rotate": (0, "w"), "drop": (0, "p"),
}
OPP_COMMANDS = {
    "left": (0, "l"), "right": (0, "r"), "down": (0, "n"),
    "rotate": (0, "u"), "drop": (0, "0"),
}

STATE_SIZE      = 4
MEM_SIZE        = 20000
BATCH_SIZE      = 64
DISCOUNT        = 0.95
EPOCHS          = 1
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_STOP_EP = 5000
REPLAY_START    = 1000
TRAIN_EPISODES  = 5000
SAVE_PATH       = "tetris_dqn.keras"


def board_features(board: np.ndarray) -> np.ndarray:
    heights = np.zeros(COLS, dtype=int)
    for col in range(COLS):
        for row in range(ROWS):
            if board[row][col]:
                heights[col] = ROWS - row
                break
    agg   = int(heights.sum())
    bump  = int(sum(abs(heights[c] - heights[c+1]) for c in range(COLS-1)))
    holes = sum(
        1 for col in range(COLS) if heights[col]
        for row in range(ROWS - heights[col], ROWS)
        if not board[row][col]
    )
    lines = int(np.all(board, axis=1).sum())
    return np.array([lines, holes, bump, agg], dtype=np.float32)


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(STATE_SIZE,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    return model


class DQNAgent:
    def __init__(self):
        self.model        = build_model()
        self.target_model = build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory        = deque(maxlen=MEM_SIZE)
        self.epsilon       = EPSILON_START
        self._decay        = (EPSILON_START - EPSILON_MIN) / EPSILON_STOP_EP
        self.update_counter = 0

    def best_action(self, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        states = np.array([board_features(a["board_result"]) for a in actions])
        qs = self.model.predict(states, verbose=0).flatten()
        return actions[int(np.argmax(qs))]

    def remember(self, state, next_board, next_piece, reward, done):
        self.memory.append((state, next_board, next_piece, reward, done))

    def train(self, pf: Pathfinder):
        if len(self.memory) < REPLAY_START:
            return
        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        x, y = [], []
        for state, next_board, next_piece, reward, done in batch:
            if done:
                target = reward
            else:
                next_actions = pf.get_actions(next_board.copy(), next_piece)
                if not next_actions:
                    max_next_q = 0
                else:
                    next_states = np.array([board_features(a["board_result"]) for a in next_actions])
                    next_qs = self.target_model.predict(next_states, verbose=0).flatten()
                    max_next_q = np.max(next_qs)
                target = reward + DISCOUNT * max_next_q
            x.append(state)
            y.append(target)
        self.model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=EPOCHS, verbose=0)
        self.update_counter += 1
        if self.update_counter % 200 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon - self._decay)

    def save(self):
        self.model.save(SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")


def play_episode(p1: Tetris, p2: Tetris, agent: DQNAgent, pf: Pathfinder):
    total_reward = 0
    while not p1.game_over and not p2.game_over:
        actions = pf.get_actions(p1.board.copy(), p1.piece)
        if not actions:
            break
        state  = board_features(p1.board.copy())
        chosen = agent.best_action(actions)
        for cmd in chosen["sequence"]:
            if cmd != "drop":
                p1.execute(cmd)
        piece = p1.piece
        for _ in range(ROWS):
            if not p1._can_move_to(piece.shape, piece.row + 1, piece.col):
                break
            piece.row += 1
        p1._place()
        pygame.event.clear()

        opp_acts = pf.get_actions(p2.board.copy(), p2.piece)
        if opp_acts:
            opp = random.choice(opp_acts)
            for cmd in opp["sequence"]:
                if cmd != "drop":
                    p2.execute(cmd)
            piece = p2.piece
            for _ in range(ROWS):
                if not p2._can_move_to(piece.shape, piece.row + 1, piece.col):
                    break
                piece.row += 1
            p2._place()
            pygame.event.clear()

        next_board = p1.board.copy()
        lines, holes, bump, height = board_features(next_board)
        reward = lines * 10 - holes * 0.5 - bump * 0.1 - height * 0.02
        if p1.game_over:
            reward = -50
        elif p2.game_over:
            reward += 20
        done = p1.game_over or p2.game_over
        agent.remember(state, next_board, p1.piece, reward, done)
        total_reward += reward

    return total_reward


def train():
    if not pygame.get_init():
        pygame.init()
        pygame.display.set_mode((1, 1))

    pf    = Pathfinder()
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
            print(f"ep={ep:4d}  reward={total_reward:7.1f}  eps={agent.epsilon:.3f}  best={best_score:.1f}")

    print("Training complete.")


if __name__ == "__main__":
    train()