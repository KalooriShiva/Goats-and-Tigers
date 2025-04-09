import random
import pickle
import matplotlib.pyplot as plt
from pullimekka import *
from functions import *
from collections import defaultdict
import copy


class GoatAI:
    def __init__(self, path='goat_q_table.pkl'):
        with open(path, 'rb') as f:
            q_data = pickle.load(f)
        self.q_table = defaultdict(float, q_data)

    def get_state(self, board_obj):
        return tuple((k, v) for k, v in board_obj.boardPositions.items())

    def get_valid_actions(self, board_obj, goats_placed):
        if goats_placed < 15:
            return [('place', pos) for pos, v in board_obj.boardPositions.items() if v == ()]
        else:
            moves = []
            for pos, v in board_obj.boardPositions.items():
                if v == 'O':
                    neighbors = Position(pos[0], pos[1]).get_neighbors()
                    for n in neighbors:
                        if Position(n[0], n[1]).content() == ():
                            moves.append(('move', (pos, n)))
            return moves

    def choose_action(self, board_obj, goats_placed):
        state = self.get_state(board_obj)
        valid_actions = self.get_valid_actions(board_obj, goats_placed)
        if not valid_actions:
            return None
        q_values = [self.q_table[(state, a)] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)


class TigerQLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.9995):
        self.q_table = defaultdict(float)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.01

    def get_state(self, board_obj):
        return tuple((k, v) for k, v in board_obj.boardPositions.items())

    def get_valid_actions(self, tiger_positions):
        actions = []
        for pos in tiger_positions:
            p = Position(pos[0], pos[1])

            # Only add captures where a goat is actually present
            captures = p.get_captures() or []
            for target in captures:
                if Position(target[0], target[1]).content() == 'O':
                    actions.append((pos, target))

            # Add normal valid moves (to empty neighbors only)
            for neighbor in p.get_neighbors():
                if Position(neighbor[0], neighbor[1]).content() == ():
                    actions.append((pos, neighbor))

        return actions


    def choose_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        q_values = [self.q_table[(state, a)] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_actions):
        max_future_q = max([self.q_table[(next_state, a)] for a in next_actions], default=0.0)
        current_q = self.q_table[(state, action)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action)] = new_q
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def save_model(self, path='tiger_q_table.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)


def simulate_tiger_move(board, action, tiger_positions, goats_captured):
    from_pos, to_pos = action

    new_board = copy.deepcopy(board)
    new_goats_captured = goats_captured
    reward = 0
    done = False

    # Set the board singleton to reflect this simulated state
    Board().boardPositions = copy.deepcopy(new_board.boardPositions)

    result = Tiger(from_pos).move(to_pos)

    if result == -1:
        reward -= 10  # Penalty for invalid move
        return board, reward, tiger_positions, done, goats_captured

    # Check if the move was a capture (this logic can be adjusted if needed)
    captures = Position(from_pos[0], from_pos[1]).get_captures()
    if captures and to_pos in captures:
        new_goats_captured += 1
        reward += 10  # Reward for capture

    # Win condition
    if new_goats_captured >= 5:
        reward += 50
        done = True

    # Update board from singleton
    new_board.boardPositions = copy.deepcopy(Board().boardPositions)
    new_tiger_positions = [to_pos if pos == from_pos else pos for pos in tiger_positions]

    return new_board, reward, new_tiger_positions, done, new_goats_captured






def plot_rewards(rewards, rolling_window=100):
    plt.figure(figsize=(10, 5))
    if len(rewards) >= rolling_window:
        smooth = [sum(rewards[i:i + rolling_window]) / rolling_window for i in range(len(rewards) - rolling_window + 1)]
        plt.plot(range(len(smooth)), smooth, label=f"Rolling Avg ({rolling_window})")
    plt.plot(rewards, alpha=0.3, label='Raw Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Tiger AI vs Goat AI - Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tiger_training_rewards.png")
    plt.show()


if __name__ == "__main__":
    agent = TigerQLearningAgent()
    goat_ai = GoatAI()
    episode_rewards = []

    for episode in range(1000):
        board = Board()
        board.clearBoard()

        tiger_positions = ['b2', 'c1', 'd2']
        for t in tiger_positions:
            Position(t[0], t[1]).place('X')

        goats_placed = 0
        goat_positions = []
        done = False
        steps = 0   
        total_reward = 0
        goats_captured = 0
        while not done and steps < 100:
            state = agent.get_state(board)
            valid_actions = agent.get_valid_actions(tiger_positions)
            if not valid_actions:
                break  # no valid tiger moves
            # print(f'Episode :{episode} , step : {steps}')
            # board.printBoard()
            action = agent.choose_action(state, valid_actions)
            new_board, reward, new_tiger_positions, done, goats_captured  = simulate_tiger_move(board, action, tiger_positions,goats_captured)
            total_reward += reward

            # Goat AI move
            goat_action = goat_ai.choose_action(new_board, goats_placed)
            if goat_action:
                if goat_action[0] == 'place':
                    pos = goat_action[1]
                    Position(pos[0], pos[1]).place('O')
                    goats_placed += 1
                    goat_positions.append(pos)
                elif goat_action[0] == 'move':
                    from_pos, to_pos = goat_action[1]
                    new_board.boardPositions[from_pos] = ()
                    new_board.boardPositions[to_pos] = 'O'
                    if from_pos in goat_positions:
                        goat_positions.remove(from_pos)
                    goat_positions.append(to_pos)


            next_state = agent.get_state(new_board)
            next_actions = agent.get_valid_actions(new_tiger_positions)
            agent.update_q(state, action, reward, next_state, next_actions)

            board = new_board
            tiger_positions = new_tiger_positions
            steps += 1

        episode_rewards.append(total_reward)

        if episode % 500 == 0:
            print(f"Episode {episode} | Epsilon: {agent.epsilon:.4f} | Reward: {total_reward}")

    agent.save_model()
    plot_rewards(episode_rewards)
