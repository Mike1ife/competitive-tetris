# Competitive Tetris

A 1v1 competitive Tetris game with DQN-trained AI agents. Players (human or AI) compete head-to-head with a garbage line attack system. Agents are trained with strategy-specific reward functions and can be evaluated in an automated tournament.

---

## Demo

Two AI agents playing against each other, with real-time combo display and garbage attacks.

---

## Setup

**Requirements:** Python ~3.12

```bash
pip install -r requirements.txt
```

Dependencies: `numpy~=2.4`, `pygame~=2.6`, `tensorflow`, `matplotlib`

---

## Usage

### Play the Game
```bash
python game.py
```
Select P1/P2 as human or AI (choose a model from `./models/`), pick difficulty, and play.

### Train an Agent
```bash
python train.py
```
Interactive prompts select strategy and opponent type. Outputs:
- Model → `./models/{strategy}/tetris_dqn_{strategy}_vs_{opponent}.keras`
- Reward curve → `./res/{strategy}/rewards_{strategy}_vs_{opponent}.png` (every 200 episodes)

### Run a Tournament
```bash
python tournament.py
```
Auto-detects all `.keras` models in `./models/` (including subfolders), runs a double round-robin, and saves results to `./res/tournament_results.csv`.

---

## AI Architecture

### Agent Pipeline
Each new piece triggers one decision cycle:
1. **Pathfinder** enumerates all valid placements (rotation × column), including hold actions
2. **Decider** scores each placement using the trained DQN model
3. The highest Q-value placement is selected and executed via hard drop

### State Vector (20 features)
```
[max_height, holes, bumpiness, lines_cleared,
 piece_one_hot(7), hold_one_hot(7),
 opponent_max_height, hold_available]
```

### DQN Training
- Architecture: 3 hidden layers (64→64→32), Huber loss, Adam optimizer
- Experience replay (buffer size 50,000), target network (updated every 200 steps)
- Epsilon-greedy exploration decaying from 1.0 → 0.05
- Opponent modes: `heuristic`, `random`, `agent` (self-play), `hybrid`
- Random garbage pre-fill at episode start for diverse board states

### Strategies
Three reward functions shape agent behavior:

| Strategy | Goal | Line rewards | Penalties |
|---|---|---|---|
| **Neutral** | Balanced | Singles penalized, Tetrises rewarded | Holes, bumpiness, height |
| **Offensive** | Maximize garbage sent | High Tetris bonus + garbage multiplier | Holes, bumpiness, height |
| **Defensive** | Survive | Reward all clears equally | Holes, bumpiness, height increase |

Death/win bonuses are applied at episode end per strategy.

---

## Game Mechanics

### Garbage System
- Attack table: Single=0, Double=1, Triple=2, Tetris=4 lines
- Combo bonus: +1 garbage per consecutive clear
- Back-to-Back Tetris: +1 bonus garbage
- Garbage cancellation: incoming lines are offset by your own attack
- Consistent single-hole column per garbage batch (Jstris-style)

### Scoring
| Clear | Points |
|---|---|
| Single | 100 |
| Double | 300 |
| Triple | 500 |
| Tetris | 800 |
| B2B Tetris | 1200 |
| Combo bonus | 50 × combo |
| Hard drop | 2 per cell |

### Hold Piece
Swap the current piece into hold (once per piece). Hold is available for both human and AI players.

### SRS Wall Kicks
Full Super Rotation System with per-piece kick tables for all rotation states.

### Difficulty (AI speed cap)
| Level | ms per piece |
|---|---|
| Easy | 1000 |
| Medium | 500 |
| Hard | 333 |

---

## Project Structure

```
competitive-tetris/
├── game.py              # Entry point, game loop, rendering
├── home.py              # Main menu UI
├── tetris.py            # Core game logic (board, pieces, gravity, garbage)
├── config.py            # Constants, tetromino shapes, SRS tables, key bindings
├── train.py             # DQN training loop
├── tournament.py        # Automated multi-model tournament with CSV export
├── agents/
│   ├── agent.py         # Top-level agent (Pathfinder + Decider)
│   ├── pathfinder.py    # Enumerate all valid placements
│   ├── decider.py       # DQN model inference, state construction
│   └── strategy.py      # Reward functions and heuristic scorer
├── models/
│   ├── defensive/       # Trained defensive models
│   ├── neutral/         # Trained neutral models
│   └── offensive/       # Trained offensive models
└── res/
    ├── defensive/       # Reward curves for defensive training runs
    ├── neutral/         # Reward curves for neutral training runs
    ├── offensive/       # Reward curves for offensive training runs
    └── tournament_results.csv
```

---

## Controls

| Action | P1 | P2 |
|---|---|---|
| Move left/right | A / D | ← / → |
| Rotate CW / CCW | W / Z | ↑ / . |
| Rotate 180° | X | , |
| Hold | Q | RShift |
| Soft drop | S | ↓ |
| Hard drop | Space | / |

---

## Reference
- Tetris AI base: https://github.com/nuno-faria/tetris-ai
