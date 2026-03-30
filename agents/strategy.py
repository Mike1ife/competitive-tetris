from config import ROWS, COLS, SCORE_TABLE


class Strategy:
    def __init__(self):
        self.strategies = {
            "neutral": self._get_neutral_reward,
            "offensive": self._get_offensive_reward,
            "defensive": self._get_defensive_reward,
        }
        self.heuristic_table = {0: 0, 1: 150, 2: 400, 3: 800, 4: 1600}

    def get_reward(
        self,
        strategy_name: str,
        normal_cleared: int,
        garbage_cleared: int,
        holes: int,
        bumpiness: int,
        height_delta: int,
    ):
        return self.strategies[strategy_name](
            normal_cleared, garbage_cleared, holes, bumpiness, height_delta
        )

    def _get_neutral_reward(
        self,
        normal_cleared: int,
        garbage_cleared: int,
        holes: int,
        bumpiness: int,
        height_delta: int,
    ):
        return (
            SCORE_TABLE.get(normal_cleared, 1200)
            + garbage_cleared * 100
            - holes * 0.75
            - bumpiness * 0.15
            - max(height_delta, 0) * 0.5
            + min(height_delta, 0) * 0.3
        )

    def _get_offensive_reward(self): ...

    def _get_defensive_reward(self): ...

    def get_heuristic(self, action: dict):
        b = action["board_result"]
        heights = [
            next((ROWS - r for r in range(ROWS) if b[r][c]), 0) for c in range(COLS)
        ]
        max_h = max(heights)
        agg_h = sum(heights)
        holes = sum(
            1
            for c in range(COLS)
            for r in range(ROWS - heights[c], ROWS)
            if not b[r][c]
        )
        bump = sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1))

        lines = action["lines_cleared"]
        line_score = self.heuristic_table.get(lines, 1600)

        return line_score - holes * 3.0 - max_h * 0.5 - agg_h * 0.1 - bump * 0.2
