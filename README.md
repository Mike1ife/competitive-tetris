# Competitive Tetris

## Reference
https://github.com/nuno-faria/tetris-ai/tree/master

---

## How the AI Works

### Step 1 — Find the Best Placement
- Try every combination of rotation × column position
- Simulate each placement on a copy of the board
- Score each resulting board using a trained DQN model:
  - **Lines cleared** (more = better)
  - **Max height** (lower = better)
  - **Holes** (fewer = better)
  - **Bumpiness** (smoother = better)
  - **Current piece** (one-hot encoded)
  - **Hold piece** (one-hot encoded)
  - **Opponent height** (higher = better for us)
  - **Hold availability**
- Pick the placement with the highest Q-value

### Step 2 — Execute the Placement
- Rotate to target orientation (with SRS wall kicks)
- Move left/right to the target column
- Hard drop straight down

---

## Design Decisions

### Down Key
The agent does not use the down key to move piece by piece. Instead, it finds the destination first, then hard drops straight down. This mirrors how a human plays — think first, then hold down.

### Dropping Rate
`FALL_INTERVAL` controls how fast pieces drop. Instead of always dropping row+1, the rate can be increased (row+rate) to represent different difficulty levels.

### Fairness — Speed Cap Mode
Since the AI decides in ~1ms (inhumanly fast), the game uses a **speed cap** approach:
- The agent is limited to placing one piece per cap interval
- Three difficulty levels control the cap:
  - **Easy** — 1000ms per piece
  - **Medium** — 500ms per piece
  - **Hard** — 333ms per piece

---

## Game Mechanics

### Garbage System (Guideline-compliant)
- Attack table: single=0, double=1, triple=2, tetris=4 garbage lines
- Total lines cleared (including garbage lines) determine attack strength
- Pending garbage queue with counter-attack cancellation
- Garbage rises from the bottom on a non-clearing placement
- Consistent single-hole garbage rows per batch (Jstris style)

### Combo System
- Consecutive line clears increment a combo counter
- Each combo level adds +1 bonus garbage on top of the base attack

### Back-to-Back
- Consecutive Tetrises send +1 bonus garbage
- Only singles, doubles, or triples break the B2B chain

### Scoring (Guideline-compliant)
- Single: 100, Double: 300, Triple: 500, Tetris: 800
- B2B Tetris: 1200 (800 + 400 bonus)
- Combo bonus: 50 × combo count
- Hard drop: 2 points per cell dropped

---

## Training
- DQN with experience replay, target network, epsilon-greedy exploration
- Three strategy-specific reward functions: neutral, offensive, defensive
- Trained against a heuristic opponent
- Garbage pre-fill at episode start for diverse board states