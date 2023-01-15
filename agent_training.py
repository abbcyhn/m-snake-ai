import torch
import random
import numpy as np
from collections import deque
from helpers_plotting import plot
from environment import SnakeGameAI
from agent_core import Linear_QNet, QTrainer

BATCH_SIZE = 1_000
MAX_MEMORY = 100_000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_iterations = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_action(self, state):
        action = [0, 0, 0]

        self.epsilon = 80 - self.n_iterations  # get random moves: tradeoff exploration / exploitation
        if random.randint(0, 200) < self.epsilon:
            direction = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model (state0)
            direction = torch.argmax(prediction).item()

        action[direction] = 1

        return action

    def remember(self, state_old, action, reward, done, state_new):
        self.memory.append((state_old, action, reward, done, state_new))

    def train_short_memory(self, state_old, action, reward, done, state_new):
        self.trainer.train(state_old, action, reward, done, state_new)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states_old, actions, rewards, dones, states_new = zip(*mini_sample)
        self.trainer.train(states_old, actions, rewards, dones, states_new)

def train():
    record = 0
    total_score = 0
    plot_scores = []
    plot_mean_scores = []

    agent = Agent()
    environment = SnakeGameAI()
    
    while True:
        # get old state
        state_old = environment.get_state()

        # get action 
        action = agent.get_action(state_old)

        # perform action and get new state
        reward, done, score = environment.play(action)
        state_new = environment.get_state()

        # train short memory
        agent.train_short_memory(state_old, action, reward, done, state_new)

        # remember
        agent.remember(state_old, action, reward, done, state_new)

        if done:
            # train long memory
            agent.n_iterations += 1
            environment.reset()
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.n_iterations, 'Score:', score, 'Record:', record)

            total_score += score
            plot_scores.append(score)
            plot_mean_scores.append(total_score/agent.n_iterations)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()