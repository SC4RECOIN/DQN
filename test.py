from agent import Agent
import torch
import numpy as np
import gym
import sys

env = gym.make('LunarLander-v2')
agent = Agent(state_size=8, action_size=4)
agent.model = torch.load('DDQN.pth')
episodes, render = 1, True
scores = []

if len(sys.argv) > 1:
    episodes = int(sys.argv[1])
    render = False

for e in range(episodes):
    state = env.reset()
    score, done = 0, False
    while not done:
        if render: env.render()
        action = agent.act(state, 0)
        state, reward, done, _ = env.step(action)
        score += reward

    scores.append(score)
    print(f'\rEpisode {e}\tScore: {scores[-1]:.2f}', end='')

if not render:  
    print(f'\nAverage over {episodes} episodes: {np.average(scores):.3f}')
