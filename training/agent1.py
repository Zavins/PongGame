import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Define the agent's deep neural network model
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQN():
    random.seed(50)
    torch.manual_seed(50)
    BATCH_SIZE = 1024
    GAMMA = 0.99
    LR = 0.0001
    TARGET_UPDATE = 100
    MEMORY_CAPACITY = 10000
    N_FEATURES = 8
    HIDDEN_SIZE = 128
    BINS = 10
    ANGLE_BINS = 2
    MODEL_PATH = 'model.pth'
    ACTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 0), 5: (0, 1), 6: (1, -1), 7: (1, 0), 8: (1, 1)}
    N_ACTIONS = len(ACTIONS)

    def __init__(self, game_step):
        self.game_step = game_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(self.N_FEATURES, self.N_ACTIONS, self.HIDDEN_SIZE).to(self.device)
        if os.path.exists(self.MODEL_PATH):
            print("loaded")
            self.load_model()
        self.target_net = QNetwork(self.N_FEATURES, self.N_ACTIONS, self.HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.LR)
        self.memory = deque(maxlen=self.MEMORY_CAPACITY)
        self.steps_done = 0
        
        self.last_player1_score = 0
        self.last_player2_score = 0
        self.last_player1_hit = 0
        self.last_player2_hit = 0
        self.prev_action = None
        self.prev_state = None
        self.prev_done = None
        self.prev_reward = None
        self.score_diff = 0
        self.total_rewards = 0

    def select_action(self, state, epsilon):
        sample = random.random()
        if sample < epsilon:
            action = random.choice(range(self.N_ACTIONS))
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = random.sample(self.memory, self.BATCH_SIZE)
        batch = list(zip(*batch))

        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.GAMMA * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def check_done(self, player1, player2):
        if player1["score"] - player2["score"] != self.score_diff:
            self.score_diff = player1["score"] - player2["score"]
            return True
        return False


    def get_reward(self, player1, player2, ball_y):
        if player1["score"] - self.last_player1_score >= 1:
            return 0
        if player1["hit"] - self.last_player1_hit >= 1:
            return 5
        if player2["score"] - self.last_player2_score >= 1:
            return -5
        
        if player2["hit"] - self.last_player2_hit >= 1:
            return 0
        if ball_y > 225:
            return -0.01
        return 0
    
    def update_stats(self, player1, player2):
        self.last_player1_score = player1["score"]
        self.last_player1_hit = player1["hit"]
        self.last_player2_score = player2["score"]
        self.last_player2_hit = player2["hit"]

    def process_data(self, data):
        player1 = data['player1']
        player2 = data['player2']
        ball = data['ball']

        player1_X = (player1['x'] // self.BINS)
        player1_angle = int(player1['angle'] // self.ANGLE_BINS)
        player2_X = (player2['x'] // self.BINS)
        player2_angle = int(player2['angle'] // self.ANGLE_BINS)
        ball_X = (ball['x'] // self.BINS)
        ball_Y = (ball['y'] // self.BINS)
        ball_velocity_X = int((ball['velocity']['x'] * 100) // 2)
        ball_velocity_Y = int((ball['velocity']['y'] * 100) // 2)
        # print(player2_X)

        state = np.array([player2_X, player2_angle, ball_X, ball_Y, ball_velocity_X, ball_velocity_Y, player1_X, player1_angle])
        reward = self.get_reward(player1, player2, ball['y'])
        done = self.check_done(player1, player2)
        self.update_stats(player1, player2)
        return state, reward, done

    def reset_stat(self):
        self.last_player1_score = 0
        self.last_player1_hit = 0
        self.last_player2_score = 0
        self.last_player2_hit = 0
        print(self.total_rewards)
        self.total_rewards = 0

    async def train(self, n_episodes=10000, epsilon_start=0.8, epsilon_end=0.01, epsilon_decay=0.99):
        epsilon = epsilon_start
        for episode in range(n_episodes):
            self.reset_stat()
            data = await self.game_step({"game": "reset"})
            state = self.process_data(data)[0]
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state, epsilon)
                data = await self.game_step({"position": self.ACTIONS[action][0], "angle": self.ACTIONS[action][1]})
                next_state, reward, done = self.process_data(data)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.optimize_model()

                if self.steps_done % self.TARGET_UPDATE == 0:
                    self.update_target_network()

                self.steps_done += 1

            if epsilon > epsilon_end:
                epsilon *= epsilon_decay

            print(f"Episode: {episode + 1}, Total reward: {total_reward}")

            if episode % 100 == 0: #save at every 100 episode
                self.save_model()

        print("Training complete.")

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.MODEL_PATH)

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.MODEL_PATH))