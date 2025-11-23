import random

import numpy as np


class TicTacToeEnv:
    def __init__(self, seed=42):
        # 0 = Empty, 1 = Player 'O' (You), -1 = Opponent 'X' (Random)
        self.rng = np.random.default_rng(seed)

        self.board = np.zeros((3, 3), dtype=int)
        self.done = False

        self.reset()

    def reset(self):
        """
        Clears the board.
        Constraint: Opponent 'X' (-1) always plays first at (1, 1).
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False

        # Opponent plays X (-1) in the center immediately
        self.board[1, 1] = -1

        return self.board.copy()

    def get_legal_moves(self):
        """Returns a list of tuples (row, col) for empty spots."""
        return list(zip(*np.where(self.board == 0)))

    def check_status(self):
        """
        Checks if the game is over.
        Returns: is_over (bool), reward (float)
        """
        # Sum of rows, cols, and diagonals
        # If sum is 3, Player 'O' (1) wins. If -3, Opponent 'X' (-1) wins.
        sums = (
            list(self.board.sum(axis=0))
            + list(self.board.sum(axis=1))
            + [np.trace(self.board), np.trace(np.fliplr(self.board))]
        )

        if 3 in sums:
            return True, 1.0  # Player wins
        if -3 in sums:
            return True, 0.0  # Bot wins
        if 0 not in self.board:
            return True, 0.0  # Draw

        return False, 0.0  # Game is not over

    def step(self, action):
        """
        1. Player 'O' places marker.
        2. Check Win.
        3. Random Opponent 'X' places marker.
        4. Check Win.
        """
        row, col = action

        if self.done:
            raise ValueError("Game is already over. Please reset.")
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move {action}. Spot occupied.")

        self.board[row, col] = 1

        is_over, reward = self.check_status()
        if is_over:
            self.done = True
            return self.board.copy(), self.done, reward

        legal_moves = self.get_legal_moves()

        if len(legal_moves) > 0:
            opp_move_idx = self.rng.integers(0, len(legal_moves))
            opp_move = legal_moves[opp_move_idx]

            self.board[opp_move] = -1

            # Check if opponent won
            is_over, reward = self.check_status()
            if is_over:
                self.done = True
                return self.board.copy(), self.done, reward

        return self.board.copy(), self.done, 0.0
