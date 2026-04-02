from config import ROWS, COLS


class Strategy:
    def __init__(self):
        self.strategies = {
            "neutral": self._get_neutral_reward,
            "offensive": self._get_offensive_reward,
            "defensive": self._get_defensive_reward,
        }
        self.heuristic_table = {0: 0, 1: 150, 2: 400, 3: 800, 4: 1600}
        self.penalties = {
            "neutral": {"death": -2000, "win": 1500},
            "offensive": {"death": -1000, "win": 2000},
            "defensive": {"death": -2000, "win": 500},
        }

    def get_reward(
        self,
        strategy_name: str,
        total_cleared: int,
        holes: int,
        bumpiness: int,
        max_height: int,
        height_delta: int,
    ):
        return self.strategies[strategy_name](
            total_cleared,
            holes,
            bumpiness,
            max_height,
            height_delta,
        )

    def _get_neutral_reward(
            self,
            total_cleared: int,
            holes: int,
            bumpiness: int,
            max_height: int,
            height_delta: int,
        ):
            line_rewards = {0: 0, 1: -10, 2: 500, 3: 1200, 4: 4000}
            attack_table = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
            garbage_send = attack_table.get(total_cleared, 4)
            return (
                line_rewards.get(total_cleared, 4000)
                + garbage_send * 200
                - holes * 3.0
                - bumpiness * 0.5
                - max(height_delta, 0) * 2.0
                - min(height_delta, 0) * 0.5
            )

    def _get_offensive_reward(
        self,
        total_cleared: int,
        holes: int,
        bumpiness: int,
        max_height: int,
        height_delta: int,
    ):
        attack_table = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}
        garbage_send = attack_table.get(total_cleared, 4)
        return (
            garbage_send * 1500
            - holes * 2.0
            - bumpiness * 0.2
            - max(height_delta, 0) * 1.0
            + min(height_delta, 0) * 0.5
        )

    def _get_defensive_reward(
        self,
        total_cleared: int,
        holes: int,
        bumpiness: int,
        max_height: int,
        height_delta: int,
    ):
        line_rewards = {0: 0, 1: 100, 2: 200, 3: 400, 4: 800}
        return (
            line_rewards.get(total_cleared, 800)
            - holes * 1.5
            - bumpiness * 0.3
            - max(height_delta, 0) * 1.0
            - min(height_delta, 0) * 0.5
        )

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