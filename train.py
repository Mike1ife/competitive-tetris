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
BATCH_SIZE = 64
MAX_PIECES = 500
DISCOUNT = 0.97
EPOCHS = 1
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_STOP_EP = 4000
REPLAY_START = 1000
TRAIN_EPISODES = 5000
TARGET_UPDATE = 200
STRATEGIES = ["neutral", "offensive", "defensive"]
OPPONENTS = ["heuristic", "random", "agent"]


class DQNAgent:
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=MEM_SIZE)
        self.epsilon = EPSILON_START
        self._decay = (EPSILON_START - EPSILON_MIN) / EPSILON_STOP_EP
        self.update_counter = 0

    def _build_model(self) -> keras.Model:
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

    def best_action(
        self,
        model: keras.Model,
        actions: list,
        piece: Piece,
        opponent_height: int,
        hold_piece: tuple = None,
        hold_used: bool = False,
        next_piece_info: tuple = None,
        explore: bool = True,
    ) -> dict:
        if explore and random.random() < self.epsilon:
            return random.choice(actions)

        states = []
        for action in actions:
            placed = _placed_piece(action, piece, hold_piece, next_piece_info)
            after_hold_piece, after_hold_used = _after_hold_state(
                action, piece, hold_piece, hold_used
            )
            states.append(
                _make_state(
                    action["board_result"],
                    action["lines_cleared"],
                    placed,
                    opponent_height,
                    after_hold_piece,
                    after_hold_used,
                )
            )
        states = np.array(states)
        qs = model(states, training=False).numpy().flatten()
        return actions[int(np.argmax(qs))]

    def remember(self, **kwargs):
        self.memory.append(kwargs)

    def train(self, pf: Pathfinder):
        if len(self.memory) < REPLAY_START:
            return
        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        x, y = [], []

        for b in batch:
            if b["done"]:
                target = b["reward"]
            else:
                next_actions = pf.get_actions(
                    b["next_board"],
                    b["next_piece"],
                    b["hold_piece"],
                    b["hold_used"],
                    b["next_piece_info"],
                )
                if not next_actions:
                    target = b["reward"]
                else:
                    next_states = []
                    for action in next_actions:
                        placed = _placed_piece(
                            action,
                            b["next_piece"],
                            b["hold_piece"],
                            b["next_piece_info"],
                        )
                        after_hold_piece, after_hold_used = _after_hold_state(
                            action, b["next_piece"], b["hold_piece"], b["hold_used"]
                        )

                        next_states.append(
                            _make_state(
                                action["board_result"],
                                action["lines_cleared"],
                                placed,
                                b["opponent_height"],
                                after_hold_piece,
                                after_hold_used,
                            )
                        )
                    next_states = np.array(next_states)
                    target = b["reward"] + DISCOUNT * np.max(
                        self.target_model(next_states, training=False).numpy().flatten()
                    )
            x.append(b["action_state"])
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


def _placed_piece(
    action: dict, piece: Piece, hold_piece: tuple, next_piece_info: tuple
) -> Piece:
    """Return the Piece that was actually placed for this action."""
    if action["sequence"] and action["sequence"][0] == "hold":
        if hold_piece is not None:
            return Piece(hold_piece[0], hold_piece[1])
        elif next_piece_info is not None:
            return Piece(next_piece_info[0], next_piece_info[1])
    return piece


def _after_hold_state(action: dict, piece: Piece, hold_piece: tuple, hold_used: bool):
    """Return the actual hold state after this action.

    action: action being taken
    piece: actual piece placed by this action
    hold_piece: info of piece been held when placing piece
    hold_used: if hold is used when placing piece
    """
    if action["sequence"] and action["sequence"][0] == "hold":
        new_hold_piece = (piece.shape, piece.color_id)
        new_hold_used = True
        return new_hold_piece, new_hold_used
    return hold_piece, hold_used


def _make_state(
    board: np.ndarray,
    lines_cleared: int,
    piece: Piece,
    opponent_height: int,
    hold_piece: tuple = None,
    hold_used: bool = False,
) -> np.ndarray:
    """Build a state vector from post-action board features.

    board: resultant board after placed piece
    lines_cleared: number of line cleared after placed piece
    piece: actual placed piece
    opponent_height: opponent's board height at the time agent decide to place piece
    hold_piece: info of piece been held after placing piece
    hold_used: if hold is used after placing piece
    """
    heights = np.zeros(COLS, dtype=int)
    for col in range(COLS):
        for row in range(ROWS):
            if board[row][col]:
                heights[col] = ROWS - row
                break
    height = int(heights.max())
    bump = int(sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1)))
    holes = sum(
        1
        for c in range(COLS)
        if heights[c]
        for r in range(ROWS - heights[c], ROWS)
        if not board[r][c]
    )
    own = np.array([height, holes, bump, lines_cleared], dtype=np.float32)

    piece_oh = np.zeros(NUM_PIECES, dtype=np.float32)
    piece_oh[piece.color_id - 1] = 1.0

    hold_oh = np.zeros(NUM_PIECES, dtype=np.float32)
    if hold_piece is not None:
        _, hold_color_id = hold_piece
        hold_oh[hold_color_id - 1] = 1.0

    hold_avail = np.array([0.0 if hold_used else 1.0], dtype=np.float32)

    return np.concatenate([own, piece_oh, hold_oh, [opponent_height], hold_avail])


def _choose_action(
    player: Tetris,
    opponent: Tetris,
    agent: DQNAgent,
    model: keras.Model,
    pf: Pathfinder,
    role: str,
    strategy: Strategy = None,
    explore: bool = True,
) -> dict:
    snap_next = player._get_next_piece_info()
    actions = pf.get_actions(
        player.board.copy(),
        player.piece,
        player.hold_piece,
        player.hold_used,
        snap_next,
    )
    if not actions:
        return None

    if role == "random":
        return random.choice(actions)
    elif role == "heuristic":
        return max(actions, key=strategy.get_heuristic)

    opponent_height = opponent.get_game_state()["max_height"]

    snap_piece = player.piece
    snap_hold = player.hold_piece
    snap_hold_used = player.hold_used

    return agent.best_action(
        model,
        actions,
        snap_piece,
        opponent_height,
        snap_hold,
        snap_hold_used,
        snap_next,
        explore,
    )


def _play_episode(
    p1: Tetris,
    p2: Tetris,
    agent: DQNAgent,
    pf: Pathfinder,
    strategy: Strategy,
    opponent_role: str,
    strategy_name: str,
    max_pieces: int = 1000,
):
    total_reward = 0
    pieces = 0
    prev_height = p1.get_game_state()["max_height"]
    while not p1.game_over and not p2.game_over and pieces < max_pieces:
        pieces += 1
        total_before = p1.normal_lines_cleared
        garbage_before = p1.garbage_lines_cleared
        snap_piece = p1.piece
        snap_hold = p1.hold_piece
        snap_next = p1._get_next_piece_info()
        p2_height = p2.get_game_state()["max_height"]

        # P1 Action
        p1_action = _choose_action(
            player=p1,
            opponent=p2,
            agent=agent,
            model=agent.model,
            pf=pf,
            role="agent",
        )

        if not p1_action:
            break

        for cmd in p1_action["sequence"]:
            p1.execute(cmd)
        pygame.event.clear()

        # P2 Action
        p2_action = _choose_action(
            player=p2,
            opponent=p1,
            agent=agent,
            model=agent.target_model,
            pf=pf,
            role=opponent_role,
            explore=False,
        )
        if p2_action:
            for cmd in p2_action["sequence"]:
                p2.execute(cmd)
            pygame.event.clear()

        normal_cleared = p1.normal_lines_cleared - total_before
        garbage_cleared = p1.garbage_lines_cleared - garbage_before
        total_cleared = normal_cleared + garbage_cleared

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

        placed = _placed_piece(p1_action, snap_piece, snap_hold, snap_next)
        action_state = _make_state(
            p1_action["board_result"],
            total_cleared,
            placed,
            p2_height,
            p1.hold_piece,
            p1.hold_used,
        )
        agent.remember(
            action_state=action_state,
            reward=reward,
            done=done,
            next_board=p1.board.copy(),
            next_piece=p1.piece.copy(),
            opponent_height=p2_height,
            hold_piece=p1.hold_piece,
            hold_used=p1.hold_used,
            next_piece_info=p1._get_next_piece_info(),
        )
        total_reward += reward

    return total_reward


def _draw_figure(rewards: list, filename: str):
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
    fig.savefig(f"./res/rewards_{filename}.png")
    plt.close(fig)


def train():
    print("Select strategy:")
    for i, name in enumerate(STRATEGIES, 1):
        print(f"  {i}. {name}")
    choice = input("Enter 1/2/3: ").strip()
    while choice not in ("1", "2", "3"):
        choice = input("Invalid. Enter 1/2/3: ").strip()

    strategy_name = STRATEGIES[int(choice) - 1]

    print("Select opponent:")
    for i, name in enumerate(OPPONENTS, 1):
        print(f"  {i}. {name}")
    choice = input("Enter 1/2/3: ").strip()
    while choice not in ("1", "2", "3"):
        choice = input("Invalid. Enter 1/2/3: ").strip()

    opponent_role = OPPONENTS[int(choice) - 1]
    save_path = f"./models/tetris_dqn_{strategy_name}_vs_{opponent_role}.keras"
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

        total_reward = _play_episode(
            p1=p1,
            p2=p2,
            agent=agent,
            pf=pf,
            strategy=strategy,
            opponent_role=opponent_role,
            strategy_name=strategy_name,
            max_pieces=MAX_PIECES,
        )

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
            _draw_figure(rewards, f"{strategy_name}_vs_{opponent_role}")

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"Training complete. Total time: {h}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    train()
