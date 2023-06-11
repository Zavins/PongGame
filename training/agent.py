import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque

# Define the agent's deep neural network model
class Policy(nn.Module):
    def __init__(self, N_FEATURES, N_ACTIONS, H_SIZE):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(N_FEATURES, H_SIZE)
        self.fc2 = nn.Linear(H_SIZE, H_SIZE)
        self.fc3 = nn.Linear(H_SIZE, H_SIZE)
        self.fc4 = nn.Linear(H_SIZE, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)

    def select_action(self, state, device, n_actions, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()

        if np.random.uniform() < epsilon:
            # Explore: select a random action
            # action = np.random.randint(0, n_actions)
            if state[0][0] > state[0][2]:
                action =np.random.randint(0, 3)
            elif state[0][0] < state[0][2]:
                action = np.random.randint(6, 9)
            else:
                action = np.random.randint(0, n_actions)
        else:
            # Exploit: select the action with the highest probability
            action = torch.argmax(probs, dim=1).item()

        m = Categorical(probs[:, :n_actions])  # Use valid range of action probabilities
        action_tensor = torch.tensor([action], dtype=torch.int64)  # Convert action to integer tensor
        log_prob = m.log_prob(action_tensor)

        return action, log_prob

class DQN():
    torch.manual_seed(50)
    BINS = 10
    ANGLE_BINS = 2
    GAMMA = 0.99
    LR = 1e-5
    N_FEATURES = 8
    H_SIZE = 256
    MODEL_PATH = 'model.pth'

    ACTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 0), 5: (0, 1), 6: (1, -1), 7: (1, 0), 8: (1, 1)}
    N_ACTIONS = len(ACTIONS)

    def __init__(self, game_step):
        self.game_step = game_step
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.steps = 0
        self.last_player1_score = 0
        self.last_player2_score = 0
        self.last_player1_hit = 0
        self.last_player2_hit = 0
        self.score_diff = 0
        self.total_rewards = 0

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

        state = np.array([player2_X, player2_angle, ball_X, ball_Y, ball_velocity_X, ball_velocity_Y, player1_X, player1_angle])
        reward = self.get_reward(player1, player2, ball['y'])
        done = self.check_done(player1, player2)
        self.update_stats(player1, player2)
        return state, reward, done

    async def reinforce(self, policy: Policy, optimizer, n_training_episodes, max_t, gamma, print_every, epsilon=1, decay=0.998):
        scores_deque = deque(maxlen=100)
        scores = []

        for i_episode in range(1, n_training_episodes+1):
            saved_log_probs = []
            rewards = []
            self.reset_stat()
            data = await self.game_step({"game": "reset"})
            state = self.process_data(data)[0]

            for t in range(max_t):
                action, log_prob = policy.select_action(state, self.device, self.N_ACTIONS, epsilon)
                saved_log_probs.append(log_prob)
                data = await self.game_step({"position": self.ACTIONS[action][0], "angle": self.ACTIONS[action][1]})
                state, reward, done  = self.process_data(data)
                rewards.append(reward)

                if done:
                    break

            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            returns = deque(maxlen=max_t)
            n_steps = len(rewards)

            for t in range(n_steps)[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(gamma * disc_return_t + rewards[t])

            eps = np.finfo(np.float32).eps.item()
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            epsilon = epsilon*decay if epsilon > 0.2 else epsilon
            if i_episode % print_every == 0:
                torch.save(policy.state_dict(), self.MODEL_PATH)
                torch.save(policy.state_dict(), "./training/models/{:.2f}.pth".format(np.mean(scores_deque)))
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), "epsilon: ", epsilon)

        return scores

    async def train(self):
        policy = Policy(self.N_FEATURES, self.N_ACTIONS, self.H_SIZE).to(self.device)
        if os.path.exists(self.MODEL_PATH):
            print("loaded")
            policy.load_state_dict(torch.load(self.MODEL_PATH))
        optimizer = optim.Adam(policy.parameters(), lr=self.LR)
        scores = await self.reinforce(policy,
                        optimizer,
                        10000, #n_training_episodes
                        10000,
                        self.GAMMA, 
                        100)

    def check_done(self, player1, player2):
        if player1["score"] - player2["score"] != self.score_diff:
            self.score_diff = player1["score"] - player2["score"]
            return True
        return False

    def get_reward(self, player1, player2, ball_y):
        if player1["score"] - self.last_player1_score >= 1:
            return -8
        if player1["hit"] - self.last_player1_hit >= 1:
            return 8
        if player2["score"] - self.last_player2_score >= 1:
            return 18
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

    def reset_stat(self):
        self.last_player1_score = 0
        self.last_player1_hit = 0
        self.last_player2_score = 0
        self.last_player2_hit = 0
        self.total_rewards = 0
