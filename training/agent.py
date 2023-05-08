
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import numpy as np
import random

# Define the agent's deep neural network model


class DQN:
    random.seed(1234)
    BATCH_SIZE = 5000
    BINS = 20
    ANGLE_BINS = 2
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 100000
    TARGET_UPDATE = 200
    MEMORY_CAPACITY = 100000
    N_FEATURES = 8

    ACTIONS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, -1), 4: (0, 0), 5: (0, 1), 6: (1, -1), 7: (1, 0), 8: (1, 1)}
    N_ACTIONS = len(ACTIONS)

    def __init__(self) -> None:
        self.policy_net = self.build_model()
        self.target_net = self.build_model()
        self.target_net.set_weights(self.policy_net.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
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

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.N_FEATURES,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.N_ACTIONS, activation='linear'))
        return model

    def select_action(self, state):
        num = np.random.uniform()
        if num < self.eps:
            action = np.random.randint(self.N_ACTIONS)
            return action
        else:
            q_values = self.policy_net.predict(np.array([state]))
            print("qvalues", q_values, self.eps, num)
            action = np.argmax(q_values)
            return action
        

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        samples = np.random.choice(len(self.memory), self.BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in samples])
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.integer)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.bool8)

        # Compute the Q values for the current state and action
        q_values = self.policy_net.predict(states)
        q_values = np.sum(q_values * tf.one_hot(actions, self.N_ACTIONS), axis=1)

        # Compute the target Q values for the next state
        target_q_values = self.target_net.predict(next_states)
        target_q_values = np.max(target_q_values, axis=1)

        # Compute the expected Q values
        expected_q_values = rewards + (1 - dones) * self.GAMMA * target_q_values

        with tf.GradientTape() as tape:
            # Compute the Q values for the current states and actions
            q_values = self.policy_net(states)
            actions_one_hot = tf.one_hot(actions, self.N_ACTIONS)
            q_values = tf.reduce_sum(actions_one_hot * q_values, axis=1)

            # Compute the loss between the expected Q values and the predicted Q values
            mse_loss = tf.keras.losses.MeanSquaredError(name="mse_loss")
            loss = mse_loss(expected_q_values, q_values)

        # Compute the gradients of the loss with respect to the model parameters
        grads = tape.gradient(loss, self.policy_net.trainable_variables)

        # Update the model parameters using the optimizer and the gradients
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        # Update the target network weights
        self._update_target_net_weights()

    def _update_target_net_weights(self):
        if self.optimizer.iterations % self.TARGET_UPDATE == 0:
            self.target_net.set_weights(self.policy_net.get_weights())

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
        # print(state)
        reward = self.get_reward(player1, player2)
        done = self.check_done(player1, player2)
        self.update_stats(player1, player2)
        return state, reward, done
    
    def check_done(self, player1, player2):
        if player1["score"] - player2["score"] != self.score_diff:
            self.score_diff = player1["score"] - player2["score"]
            return True
        return False

    def get_reward(self, player1, player2):
        #AI score
        if player1["score"] - self.last_player1_score >= 1:
            #lost ball, -20
            return -5000
        
        if player1["hit"] -  self.last_player1_hit >= 1:
            #oppo_hit, -2
            return 100
        
        if player2["score"] - self.last_player2_score >= 1:
            #goal + 50
            return 10000
        
        if player2["hit"] - self.last_player2_hit >= 1:
            # hit + 1
            return 5000
        
        return -1

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

    def train(self, data):
        state, reward, done = self.process_data(data)
        self.total_rewards += reward
        if (self.steps !=0):
            self.remember(self.prev_state, self.prev_action, self.prev_reward, state, self.prev_done)
            if self.prev_done == True:
                self.optimize_model()
            self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-self.steps / self.EPS_DECAY)
            if self.prev_done:
                self.reset_stat()
            if self.steps % 10 == 0:
                print("Step:", self.steps)
            
        # next_state, , done, info = 
        action = int(self.select_action(state))
        self.prev_action = action
        self.prev_state = state
        self.prev_done = done
        self.prev_reward = reward
        self.steps += 1
        return {"position": self.ACTIONS[action][0], "angle": self.ACTIONS[action][1]}