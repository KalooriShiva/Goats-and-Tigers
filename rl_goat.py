from collections import defaultdict
import random
import pickle
import copy
from pullimekka import *
from functions import *
import numpy as np


def evaluate_goat_reward(board, prev_goat_count, current_goat_count, goats_placed, goats_captured, previous_board=None):
    """
    Calculates reward based on how goat performed this turn.
    """
    reward = 0

    # 🐐 Goat got captured
    if current_goat_count < prev_goat_count:
        reward -= 20

    # 🛡️ Goat survived
    if current_goat_count == prev_goat_count:
        reward += 2

    # 🎯 All goats placed
    if goats_placed == 15:
        reward += 5

    # 🏆 Win condition: Tiger is trapped
    tiger_moves = 0
    for pos, val in board.boardPositions.items():
        if val == 'X':
            neighbors = Position(pos[0], pos[1]).get_neighbors() or []
            captures = Position(pos[0], pos[1]).get_captures() or []
            tiger_moves += len(neighbors) + len(captures)

    if tiger_moves == 0:
        reward += 50

    # 😵 If goat is too passive (same count, no tiger threat), penalize slightly
    if goats_placed == 15 and current_goat_count == prev_goat_count and tiger_moves > 6:
        reward -= 2
    if previous_board:
        prev_goat_positions = {k for k, v in previous_board.boardPositions.items() if v == "O"}
        curr_goat_positions = {k for k, v in board.boardPositions.items() if v == "O"}
        moved_goat = list(curr_goat_positions - prev_goat_positions)
        if moved_goat:
            goat_pos = moved_goat[0]  # Assuming only one goat moves per turn
            safe = True
            for pos, val in board.boardPositions.items():
                if val == "X":
                    captures = Position(pos[0], pos[1]).get_captures() or []
                    if goat_pos in captures:
                        safe = False
                        break
            if safe:
                reward += 10  # Escaped threat

    return reward

def simulate_goat_move(board, action, goat_positions, num_goats_placed, num_goats_captured):
    """
    Simulates a goat's move or placement and returns the updated board, reward, 
    new goat positions, updated count of goats placed and captured, and done flag.
    """
    import copy
    new_board = copy.deepcopy(board)
    from_pos, to_pos = action
    done = False
    reward = 0

    new_goat_positions = goat_positions.copy()
    new_goats_placed = num_goats_placed
    new_goats_captured = num_goats_captured

    # 🐐 Simulate goat move
    if from_pos == "PLACE":
        if Position(to_pos[0], to_pos[1]).content() == ():
            Board().boardPositions = copy.deepcopy(new_board.boardPositions)
            result = Goat(to_pos).place()
            if result == 1:
                new_goat_positions.append(to_pos)
                new_goats_placed += 1
    else:
        Board().boardPositions = copy.deepcopy(new_board.boardPositions)
        result = Goat(from_pos).move(to_pos)
        if result == 1:
            new_goat_positions.remove(from_pos)
            new_goat_positions.append(to_pos)

    # ✅ Now sync updated global board back to new_board
    new_board.boardPositions = copy.deepcopy(Board().boardPositions)

    # 🐯 Tiger's turn to react
    goats_before = len([pos for pos in new_board.boardPositions if new_board.boardPositions[pos] == 'O'])
    goats_after = goats_before
    tigers = [pos for pos in new_board.boardPositions if new_board.boardPositions[pos] == 'X']

    tiger_acted = False
    for t in tigers:
        Board().boardPositions = copy.deepcopy(new_board.boardPositions)
        captures = Position(t[0], t[1]).get_captures()
        if captures:
            for cap in captures:
                if Tiger(t).capture(cap):
                    goats_after -= 1
                    new_goats_captured += 1
                    tiger_acted = True
                    break
        if tiger_acted:
            break

    if not tiger_acted:
        for t in tigers:
            Board().boardPositions = copy.deepcopy(new_board.boardPositions)
            neighbors = Position(t[0], t[1]).get_neighbors()
            for n in neighbors:
                if Position(n[0], n[1]).content() == ():
                    Tiger(t).move(n)
                    tiger_acted = True
                    break
            if tiger_acted:
                break

    # ✅ Update new_board after tiger moved
    new_board.boardPositions = copy.deepcopy(Board().boardPositions)

    # 🎯 Assign reward
    reward = 10 if goats_after == goats_before else -10

    if new_goats_captured >= 5:
        reward -= 20
        done = True

    # 🛑 Check for stalemate
    tiger_moves = 0
    for t in tigers:
        neighbors = Position(t[0], t[1]).get_neighbors() or []
        captures = Position(t[0], t[1]).get_captures() or []
        tiger_moves += len(neighbors) + len(captures)

    if tiger_moves == 0:
        reward += 20
        done = True

    return new_board, reward, new_goat_positions, new_goats_placed, new_goats_captured, done



class GoatQLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.999):
        self.q_table = defaultdict(float)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.01

    def get_state(self, board_obj, goats_placed, goats_captured):
        # Counts only, not full board positions
        tiger_count = len([v for v in board_obj.boardPositions.values() if v == 'X'])
        goat_count = len([v for v in board_obj.boardPositions.values() if v == 'O'])
        return (goat_count, tiger_count, goats_placed, goats_captured)


    def get_valid_actions(self, goat_positions, goats_placed):
        actions = []
        if goats_placed < 15:
            empty = emptyPositions(Board().boardPositions)
            for e in empty:
                actions.append(("PLACE", e))
        else:
            for g in goat_positions:
                for move in Goat(g).possibleMoves():
                    actions.append((g, move))
        return actions

    def choose_action(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.q_table[(state, a)] for a in valid_actions]
            max_q = max(q_values, default=0.0)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_actions):
        max_future_q = max([self.q_table[(next_state, a)] for a in next_actions], default=0.0)
        current_q = self.q_table[(state, action)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action)] = new_q
        if self.epsilon > self.min_epsilon:
            self.epsilon -= 0.00005  # adjust rate
            self.epsilon = max(self.epsilon, self.min_epsilon)


    def save_model(self, path='goat_q_table.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, path='goat_q_table.pkl'):
        with open(path, 'rb') as f:
            self.q_table = defaultdict(float, pickle.load(f))
def random_tiger_positions():
    all_positions = list(Board().boardPositions.keys())
    empty_positions = [pos for pos in all_positions if Board().boardPositions[pos] == ()]
    return random.sample(empty_positions, 3)

if __name__ == "__main__":
    agent = GoatQLearningAgent()

    import matplotlib.pyplot as plt

    # Track rewards
    reward_history = []

    for episode in range(10000):
        board = Board()
        board.clearBoard()


        tiger_positions = random_tiger_positions()

        for t in tiger_positions:
            Position(t[0], t[1]).place('X')

        goat_positions = []
        goats_placed = 0
        goats_captured = 0
        done = False
        steps = 0
        total_reward = 0  # Track episode reward
        shiva =0
        while not done:
            state = agent.get_state(board, goats_placed, goats_captured)
            valid_actions = agent.get_valid_actions(goat_positions, goats_placed)

            if not valid_actions:
                break

            action = agent.choose_action(state, valid_actions)
            
            # print(f'Episode :{episode} , Step :{steps}' )
            # board.printBoard()
            prev_goat_count = len([pos for pos in board.boardPositions if board.boardPositions[pos] == 'O'])
            previous_board = copy.deepcopy(board)
            new_board, _, new_goat_positions, new_goats_placed, new_goats_captured, done = simulate_goat_move(
            board, action, goat_positions, goats_placed, goats_captured)

            current_goat_count = len([pos for pos in new_board.boardPositions if new_board.boardPositions[pos] == 'O'])

            reward = evaluate_goat_reward(new_board, prev_goat_count, current_goat_count, new_goats_placed, new_goats_captured,previous_board)


            next_state = agent.get_state(new_board, new_goats_placed, new_goats_captured)
            next_actions = agent.get_valid_actions(new_goat_positions, new_goats_placed)

            agent.update_q(state, action, reward, next_state, next_actions)

            board = new_board
            goat_positions = new_goat_positions
            goats_placed = new_goats_placed
            goats_captured = new_goats_captured

            total_reward += reward
            steps += 1
            if steps > 100:
                break

        reward_history.append(total_reward)  # Log reward
        if episode % 500 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, epsilon: {agent.epsilon:.3f}")
        # Save model after training
    agent.save_model()

    # Plot episode vs reward
    plt.figure(figsize=(12, 6))
    rolling = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(rolling, label="Rolling Avg (100)")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Goat Agent Training: Episode vs Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
