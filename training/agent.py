# import random

# async def receive_state(data):
#     print(data)
#     position = random.choice([-1, 0, 1])
#     angle = random.choice([-1, 0, 1])
#     return {"position": position, "angle": angle}


# import tensorflow as tf
# import numpy as np
# import json

# # Define the agent's deep neural network model
# class Model(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense1 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense3 = tf.keras.layers.Dense(9, activation='softmax')

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         x = self.dense3(x)
#         return x

# # Create an instance of the agent's neural network model
# model = Model()

# # Define the agent's action selection function
# def select_action(state):
#     # Convert the state to a numpy array
#     state = np.array(state)

#     # Use the agent's neural network to predict the probabilities of each action
#     action_probs = model.predict(state.reshape((1, -1)))

#     # Sample an action from the predicted probabilities
#     action = np.random.choice(range(9))

#     # Map the action index to a specific action in the action space
#     action_space = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
#     action = action_space[action]

#     return action


# async def receive_state(state):
#     player1 = state['player1']
#     player2 = state['player2']
#     ball = state['ball']
    
#     # Preprocess the state and convert it to a numpy array
#     state = np.array([player1['x'], player1['angle'], player1['score'], player1['hit'],
#                       player2['x'], player2['angle'], player2['score'], player2['hit'],
#                       ball['x'], ball['y'], ball['radius'], ball['speed'], ball['serve'], ball['velocity']['x'], ball['velocity']['y']])
#     state = state.astype(np.float32) # cast the state to float data type
    
#     action = select_action(state)
    
#     return {"position": action[0], "angle": action[1]}


import random
import tensorflow as tf
import numpy as np
from collections import deque

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32

# Replay memory
memory = deque(maxlen=2000)

# Define the agent's deep neural network model
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(9, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Create an instance of the agent's neural network model
model = Model()

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# Define the agent's action selection function
def select_action(state):
    global epsilon
    if np.random.rand() <= epsilon:
        # Explore: select a random action
        return random.randrange(9)
    else:
        # Exploit: select the action with the highest predicted reward
        return np.argmax(model.predict(state.reshape((1, -1))))

# Define the training function
def train_model():
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        
        target = model.predict(state.reshape(1, -1))
        if done:
            target[0][action] = reward
        else:
            t = model.predict(next_state.reshape(1, -1))
            target[0][action] = reward + gamma * np.amax(t)
        with tf.GradientTape() as tape:
            predictions = model(state.reshape(1, -1))
            loss = loss_fn(target, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay






def get_reward(state, next_state):
    # Calculate the distance between player 2 and the ball in the current and next state
    current_distance = abs(state['player2']['x'] - state['ball']['x'])
    next_distance = abs(next_state['player2']['x'] - next_state['ball']['x'])

    # Calculate the reward based on the change in distance
    # If the distance is decreasing (i.e., the player is getting closer to the ball), the reward is positive
    # Otherwise, the reward is negative
    distance_reward = (current_distance - next_distance) / 100.0  # Normalizing reward by dividing by 100

    # Calculate the reward based on the score
    # If the score has increased, the reward is positive, if it has decreased, the reward is negative
    score_reward = next_state['player1']['score'] - state['player1']['score']

    # If the player missed the ball, give a large negative reward
    miss_penalty = -10.0 if next_state['player1']['hit'] == 0 and state['player1']['hit'] == 1 else 0.0

    # Sum the rewards
    reward = distance_reward + score_reward + miss_penalty

    return reward

last_state = None
last_action = None

# The action space
action_space = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

async def receive_state(state):
    global last_state, last_action, memory

    player1 = state['player1']
    player2 = state['player2']
    ball = state['ball']

    # Preprocess the state and convert it to a numpy array
    state_array = np.array([player1['x'], player1['angle'], player1['score'], player1['hit'],
                            player2['x'], player2['angle'], player2['score'], player2['hit'],
                            ball['x'], ball['y'], ball['radius'], ball['speed'], ball['serve'], 
                            ball['velocity']['x'], ball['velocity']['y']])
    state_array = state_array.astype(np.float32)  # cast the state to float data type

    if last_state is not None and last_action is not None:
        # Calculate the reward
        reward = get_reward(last_state, state)

        # Check if the game has ended (player1 or player2 scored a point)
        done = state['player1']['score'] > last_state['player1']['score'] or state['player2']['score'] > last_state['player2']['score']


        # Save the experience in the replay memory
        memory.append((last_state, last_action, reward, state_array, done))


        # Train the model
        train_model()

    action = select_action(state_array)

    # Save the current state and action
    last_state = state
    last_action = action

    # Map the action index to a specific action in the action space
    action = action_space[action]

    return {"position": action[0], "angle": action[1]}
