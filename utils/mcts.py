import numpy as np

from .environment import TicTacToeEnv


class MCTSNode:
    def __init__(self, state_key: tuple, reward: int):
        self.state_key = state_key
        self.q_values = {}  # Q(x, a)
        self.action_counts = {}  # n(x, a) - denominator
        self.transition_counts = {}  # n(x' | x, a) - numerator term
        self.reward = reward


class PureRandomMCTS:
    def __init__(self, rng, gamma=1.0):
        self.gamma = gamma
        self.nodes = {}  # Map state_hash -> MCTSNode
        self.rng = rng

    def reset(self):
        self.nodes = {}

    def get_state_key(self, state: np.ndarray) -> tuple:
        """
        Returns a 'hashable' key of the state (board).
        Right now, we turn it into a tuple of tuples.
        """
        return tuple(map(tuple, state))

    def get_node(self, state: np.ndarray, reward: int = 0) -> MCTSNode:
        """
        Creates and returns a state node.
        Returns a cached one if it already exists.
        """
        state_key = self.get_state_key(state)
        if state_key not in self.nodes:
            self.nodes[state_key] = MCTSNode(state_key, reward)
        return self.nodes[state_key]

    def clone_env(self, env: TicTacToeEnv) -> TicTacToeEnv:
        """
        Clones the current environment
        """
        return TicTacToeEnv(board=env.board, rng=self.rng)

    def search(
        self, initial_env: TicTacToeEnv, n_iterations=1000, track_q_value_history=False
    ) -> tuple:
        """
        Performs Pure Random MCTS search from the given initial environment state.
        Return the best action tuple found.
        """
        root = self.get_node(initial_env.board)
        root_q_value_history = {}

        for _ in range(n_iterations):
            # --- 1. SAMPLE TRAJECTORY ---
            # We start from the root and play a random game.
            # We then keep track of the path, which is represented as a list of tuples:
            # [(node, action, next_node), ...]

            path = []
            env = self.clone_env(initial_env)

            while not env.done:
                legal_moves = env.get_legal_moves()

                if not legal_moves:
                    break

                # Pure Random Action
                random_action_idx = self.rng.integers(0, len(legal_moves))
                action = legal_moves[random_action_idx]

                # Execute step
                prev_node = self.get_node(env.board)
                next_state, is_done, reward = env.step(action)

                # Record the transition
                path.append(
                    (
                        prev_node,
                        action,
                        self.get_node(next_state, reward),
                    )
                )

            # --- 2. UPDATE ESTIMATES (Back up from leaf to root) ---
            # We iterate backwards through the path we just took
            for step in reversed(path):
                node, action, next_node = step

                # Update Counts
                # n(x, a) += 1
                node.action_counts[action] = node.action_counts.get(action, 0) + 1

                # n(x' | x, a) += 1
                t_key = (action, next_node.state_key)
                node.transition_counts[t_key] = node.transition_counts.get(t_key, 0) + 1

                # CALCULATE NEW Q(x, a) using the formula
                # Q(x,a) = Sum_x' [ n(x'|x,a) * (r + gamma * max_a' Q(x', a')) ] / n(x,a)

                # 1. Start calculating the weighted sum (the nominator)
                # by iterating over all SEEN next states x' at this (x,a) step in the path
                relevant_transitions = [
                    (k[1], count)
                    for k, count in node.transition_counts.items()
                    if k[0] == action
                ]

                nominator_sum = 0
                for relevant_next_state_key, count in relevant_transitions:
                    outcome_node = self.nodes[relevant_next_state_key]

                    outcome_reward = outcome_node.reward
                    max_next_q = max(outcome_node.q_values.values(), default=0)

                    nominator_sum += count * (outcome_reward + self.gamma * max_next_q)

                node.q_values[action] = nominator_sum / node.action_counts[action]

            if track_q_value_history:
                for action, q_value in root.q_values.items():
                    if action not in root_q_value_history:
                        root_q_value_history[action] = []
                    root_q_value_history[action].append(q_value)

        # Return best action + history of its Q-values (if tracked)
        best_action = max(root.q_values, key=root.q_values.get)
        return best_action, root_q_value_history.get(best_action, [])
