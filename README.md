# Competitive Tetris

## Reference
https://github.com/nuno-faria/tetris-ai/tree/master

---

## How the AI Works

### Step 1 — Find the Best Placement
- Try every combination of rotation × column position
- Simulate each placement on a copy of the board
- Score each resulting board using heuristics:
  - **Lines cleared** (more = better)
  - **Aggregate height** (lower = better)
  - **Holes** (fewer = better)
  - **Bumpiness** (smoother = better)
- Pick the best placement

### Step 2 — Execute the Placement
- Move left/right to the target column
- Hard drop straight down
- **T-spin (TODO)**: drop to a certain height, rotate at the last moment to slide into a hole

---

## Design Decisions

### Down Key
The agent does not use the down key to move piece by piece. Instead, it finds the destination first, then hard drops straight down. This mirrors how a human plays — think first, then hold down.

### Dropping Rate
`FALL_INTERVAL` controls how fast pieces drop. Instead of always dropping row+1, the rate can be increased (row+rate) to represent different difficulty levels.

### Fairness — Turn-Based Mode
Since the AI decides in ~1ms (inhumanly fast), the game uses a **turn-based** approach:
- The bot doesn't receive the next piece until the human places the same piece first
- Both players always work with the same piece sequence
- This makes it a fair comparison of decision-making, not reaction speed

### Difficulty Levels
Three difficulty levels based on a **decision cap** (pieces per second):
- **Easy** — slow cap
- **Medium** — medium cap  
- **Hard** — fast cap (near full AI speed)

---

## Current Implementation Notes
- Current implementation is solely used for PvP (no AI agent yet)
- The `execute()` method is set up for agent use, similar to hw1
- Score calculation should be refined so clearing more lines together gets a bonus
- Garbage lines contain 1–3 holes (may be modified in the future)
- Some bugs may exist regarding garbage lines

---

## TODO
- [ ] Build heuristic evaluation function (holes, height, bumpiness, lines cleared)
- [ ] Generate all possible placements for a given piece
- [ ] Execute chosen placement via `execute()` commands
- [ ] Implement turn-based / speed cap logic in `game.py`
- [ ] T-spin support (drop to hole, rotate at last moment)
- [ ] Refine score calculation (bonus for multi-line clears)
