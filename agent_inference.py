import torch
import numpy as np
from agent_core import Linear_QNet
from environment import SnakeGameAI

def get_action(model, state):
    action = [0, 0, 0]

    model.eval()
    with torch.no_grad():
        state0 = torch.tensor(state, dtype = torch.float)
        prediction = model (state0)
        direction = torch.argmax(prediction).item()

    action[direction] = 1
    
    return action


if __name__ == "__main__":
    # init
    done = False
    environment = SnakeGameAI()

    # load back the model
    model = Linear_QNet(11, 256, 3)
    state_dict = torch.load("agent_model.pth")
    model.load_state_dict(state_dict)

    while not done:
        # get state
        state = environment.get_state()

        # get action 
        action = get_action(model, state)

        # perform action and get new state
        reward, done, score = environment.play(action)
    
    print("Score:", score)