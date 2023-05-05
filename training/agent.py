# import random

# async def receive_state(data):
#     print(data)
#     position = random.choice([-1, 0, 1])
#     angle = random.choice([-1, 0, 1])
#     return {"position": position, "angle": angle}


import tensorflow as tf
import numpy as np
import json

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

# Define the agent's action selection function
def select_action(state):
    # Convert the state to a numpy array
    state = np.array(state)

    # Use the agent's neural network to predict the probabilities of each action
    action_probs = model.predict(state.reshape((1, -1)))

    # Sample an action from the predicted probabilities
    action = np.random.choice(range(9), p=action_probs.ravel())

    # Map the action index to a specific action in the action space
    action_space = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    action = action_space[action]

    return action


async def receive_state(state):
    state_data = state['text']
    state_dict = json.loads(state_data)
    
    player1 = state_dict['player1']
    player2 = state_dict['player2']
    ball = state_dict['ball']
    # preprocess the state and convert it to a numpy array
    state = np.array([player1['x'], player1['angle'], player1['score'], player1['hit'],
                      player2['x'], player2['angle'], player2['score'], player2['hit'],
                      ball['x'], ball['y'], ball['radius'], ball['speed'], ball['serve'], ball['velocity']['x'], ball['velocity']['y']])
    state = state.astype(np.float32) # cast the state to float data type
    action = select_action(state)
    return {"position": action[0], "angle": action[1]}


