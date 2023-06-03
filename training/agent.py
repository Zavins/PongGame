import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

# Define the agent's deep neural network model
class Policy(nn.Module):

    def __init__(self, N_FEATURES, N_ACTIONS, H_SIZE):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(N_FEATURES, H_SIZE)
        self.fc2 = nn.Linear(H_SIZE, H_SIZE)
        self.fc3 = nn.Linear(H_SIZE, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def select_action(self, state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        # print(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class DQN():
    # random.seed(1234)
    # BATCH_SIZE = 1024
    torch.manual_seed(50)
    BINS = 10
    ANGLE_BINS = 2
    GAMMA = 0.99
    LR = 1e-5
    # TARGET_UPDATE = 200
    # MEMORY_CAPACITY = 100000
    N_FEATURES = 8
    H_SIZE = 128
    MODEL_PATH = 'model.pth'

    ACTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 0), 5: (0, 1), 6: (1, -1), 7: (1, 0), 8: (1, 1)}
    N_ACTIONS = len(ACTIONS)

    def __init__(self, game_step):
        self.game_step = game_step
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.policy = Policy(self.N_FEATURES).to(self.device)
        # self.target_net = Policy(self.N_FEATURES).to(self.device)
        # self.target_net.load_state_dict(self.policy.state_dict())
        # self.target_net.eval()
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=self.LR)
        # self.memory = deque(maxlen=self.MEMORY_CAPACITY)
        self.steps = 0
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

    async def reinforce(self, policy:Policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
        scores_deque = deque(maxlen=100)
        scores = []
        # Line 3 of pseudocode
        for i_episode in range(1, n_training_episodes+1):
            saved_log_probs = []
            rewards = []
            self.reset_stat()
            data = await self.game_step({"game": "reset"})
            state = self.process_data(data)[0]
            # Line 4 of pseudocode
            for t in range(max_t):
                action, log_prob = policy.select_action(state, self.device)
                saved_log_probs.append(log_prob)
                data = await self.game_step({"position": self.ACTIONS[action][0], "angle": self.ACTIONS[action][1]})
                state, reward, done  = self.process_data(data)
                rewards.append(reward)
                if done:
                    break 
            print(sum(rewards))
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
        
            # Line 6 of pseudocode: calculate the return
            returns = deque(maxlen=max_t) 
            n_steps = len(rewards) 
        
            for t in range(n_steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)
                returns.appendleft( gamma * disc_return_t + rewards[t]   ) # Code Here: complete here        
       
            ## standardization for training stability
            eps = np.finfo(np.float32).eps.item()
            
            ## eps is added to the standard deviation of the returns to avoid numerical instabilities
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            
            # Line 7:
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()
            
            # Line 8: PyTorch prefers gradient descent 
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            if i_episode % print_every == 0:
                torch.save(policy.state_dict(), self.MODEL_PATH)
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
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





    # def optimize_model(self):
    #     if len(self.memory) < self.BATCH_SIZE:
    #         return

    #     batch = random.sample(self.memory, self.BATCH_SIZE)
    #     states, actions, rewards, next_states, dones = zip(*batch)

    #     states = torch.tensor(states, dtype=torch.float, device=self.device)
    #     actions = torch.tensor(actions, dtype=torch.long, device=self.device)
    #     rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
    #     next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
    #     dones = torch.tensor(dones, dtype=torch.float, device=self.device)

    #     q_values = self.policy.forward(states)
    #     q_values = torch.sum(q_values * F.one_hot(actions, self.N_ACTIONS), dim=1)

    #     with torch.no_grad():
    #         target_q_values = self.target_net(next_states)
    #         target_q_values = torch.max(target_q_values, dim=1).values

    #     expected_q_values = rewards + (1 - dones) * self.GAMMA * target_q_values

    #     loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     self._update_target_net_weights()

    # def _update_target_net_weights(self):
    #     if self.steps % self.TARGET_UPDATE == 0:
    #         self.target_net.load_state_dict(self.policy.state_dict())

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))



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

    def reset_stat(self):
        self.last_player1_score = 0
        self.last_player1_hit = 0
        self.last_player2_score = 0
        self.last_player2_hit = 0
        print(self.total_rewards)
        self.total_rewards = 0

    # def train(self, data):
    #     state, reward, done = self.process_data(data)
    #     self.total_rewards += reward
    #     if self.steps != 0:
    #         self.remember(self.prev_state, self.prev_action, self.prev_reward, state, self.prev_done)
    #         if self.prev_done:
    #             self.optimize_model()
    #             self.reset_stat()
    #         if self.steps % 10 == 0:
    #             print("Step:", self.steps)

    #     action, p = self.policy.select_action(state, self.device)
    #     self.prev_action = action
    #     self.prev_state = state
    #     self.prev_done = done
    #     self.prev_reward = reward
    #     self.steps += 1
    #     action = int(action)
    #     return 
