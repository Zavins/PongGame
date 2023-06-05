import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN:
    BATCH_SIZE = 10000
    BINS = 20
    ANGLE_BINS = 2
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.003
    EPS_DECAY = 0.99999999
    TARGET_UPDATE = 200
    MEMORY_CAPACITY = 1000000
    N_FEATURES = 8
    TOP_MODELS_COUNT = 5

    

    ACTIONS = {0: (-2, -2), 1: (-2, 0), 2: (-2, 2), 3: (0, -2), 4: (0, 0), 5: (0, 2), 6: (2, -2), 7: (2, 0), 8: (2, 2)}
    # ACTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 0), 5: (0, 1), 6: (1, -1), 7: (1, 0), 8: (1, 1)}
    # ACTIONS = {0: (-3, -3), 1: (-3, 0), 2: (-3, 3), 3: (0, -3), 4: (0, 0), 5: (0, 3), 6: (3, -3), 7: (3, 0), 8: (3, 3)}
    N_ACTIONS = len(ACTIONS)

    def __init__(self) -> None:
        self.policy_net = self.build_model().to(device)
        self.target_net = self.build_model().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-2)
        self.memory = []
        self.eps = self.EPS_START
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
        self.progress = 0
        self.progresslist = []
        self.top_models = [{"reward": float('-inf'), "file_path": None} for _ in range(self.TOP_MODELS_COUNT)]
        self.reward_threshold = float('-inf')

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.N_FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.N_ACTIONS)
        )
        return model

    def select_action(self, state):
        num = np.random.uniform()
        if num < self.eps:
            # print("----random action------")
            action = np.random.randint(self.N_ACTIONS)
            return action
        else:
            # print("----AI moving------")
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()
            return action

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        samples = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32).to(device)

        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + (1 - dones.float()) * self.GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        # Add L1 regularization
        l1_lambda = 0.001  # Regularization factor
        l1_reg = torch.tensor(0.).to(device)
        for param in self.policy_net.parameters():
            l1_reg += torch.norm(param, p=1)
        loss += l1_lambda * l1_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


            if self.steps % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.MEMORY_CAPACITY:
            self.memory.pop(0)

    def process_data(self, data):
        player1 = data['player1']
        player2 = data['player2']
        ball = data['ball']

        player1_X = (player1['x'] // self.BINS)
        player1_angle = int(player1['angle']//self.ANGLE_BINS)
        player2_X = (player2['x'] //  self.BINS)
        player2_angle = int(player2['angle']//self.ANGLE_BINS)
        ball_X = (ball['x'] //  self.BINS)
        ball_Y = (ball['y']// self.BINS)
        ball_velocity_X = int((ball['velocity']['x']*100)//5)
        ball_velocity_Y = int((ball['velocity']['y']*100)//5)

        state = np.array([player1_X, player1_angle, player2_X, player2_angle, ball_X, ball_Y, ball_velocity_X, ball_velocity_Y])
        done = self.check_done(player1, player2)
        reward = self.get_reward(player1, player2, done)
        self.update_stats(player1, player2)
        return state, reward, done

    def check_done(self, player1, player2):
        if player1["score"] - player2["score"] != self.score_diff:
            self.score_diff = player1["score"] - player2["score"]
            return True
        return False

    def get_reward(self, player1, player2, done):
        reward = 0

        # If game ended (someone scored), give a large negative reward if the AI lost the round.
        if done:
            if player2["score"] > self.last_player2_score:
                reward += 1000
            # if player1["score"] > self.last_player1_score:
            #     reward -= 100
        else: # game hasn't ended, check for beneficial actions
            if player2["hit"] > self.last_player2_hit: # AI hit the ball
                reward += 100

        # Update player statistics
        self.update_stats(player1, player2)

        # accumulate progress
        self.progress += reward
        return reward


    

    def update_stats(self, player1, player2):
        self.last_player1_score = player1["score"] 
        self.last_player1_hit = player1["hit"]
        self.last_player2_score = player2["score"]
        self.last_player2_hit = player2["hit"] 

    def reset_stat(self):
        # self.last_player1_score = 0
        # self.last_player1_hit = 0
        # self.last_player2_score = 0
        # self.last_player2_hit = 0
        # print("reset")
        return 
        

    def train(self, data):
        state, reward, done = self.process_data(data)
        self.total_rewards += reward
        if (self.steps !=0):
            self.remember(self.prev_state, self.prev_action, self.prev_reward, state, self.prev_done)
            if self.steps % 1000 == 0:
                self.optimize_model()
            self.eps = self.eps*self.EPS_DECAY if self.eps>self.EPS_END else self.eps
            
            if self.steps % 1000 == 0:
                print("Step: ", self.steps, reward)
                print("EPS: ",self.eps)
                print("Progress reward: ", self.progress)
                print("--------------")
                self.progresslist.append(self.progress)
                self.progress = 0
                
            if self.steps % 10000 == 0:  # Save model every 10000 steps
                model_path = f'may3_step_{self.steps}.pth'
                self.save_model(self.total_rewards/10000, model_path)
                self.total_rewards = 0

            if self.steps % 3000000 == 0:  # Save model every 10000 steps
                self.plot_progress()

            if self.prev_done:
                self.reset_stat()

            
        action = int(self.select_action(state))
        self.prev_action = action
        self.prev_state = state
        self.prev_done = done
        self.prev_reward = reward
        self.steps += 1
        return {"position": self.ACTIONS[action][0], "angle": self.ACTIONS[action][1]}


    def save_model(self, reward, file_path):
        if reward > self.reward_threshold:
            # Remove the model with the lowest reward
            min_reward_model = min(self.top_models, key=lambda x: x['reward'])
            self.top_models.remove(min_reward_model)

            # If a model file exists for the removed model, delete it
            if min_reward_model['file_path'] is not None and os.path.exists(min_reward_model['file_path']):
                os.remove(min_reward_model['file_path'])

            # Save new model and add its information to the top_models list
            torch.save(self.policy_net.state_dict(), file_path)
            self.top_models.append({"reward": reward, "file_path": file_path})

            # Update reward_threshold
            self.reward_threshold = min(self.top_models, key=lambda x: x['reward'])['reward']

    def load_model(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def plot_progress(self):
        step = int((len(self.progresslist))/10)
        print(step)
        avg_rewards = [sum(self.progresslist[i:i+step])/step for i in range(0, len(self.progresslist), step)]
        
        plt.plot(avg_rewards)
        print(avg_rewards)
        plt.title('Average training reward over time')
        plt.ylabel('Average Reward')
        plt.xlabel('Steps (in 10 thousands)')
        plt.show()
