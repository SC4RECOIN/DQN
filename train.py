from agent import Agent
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from collections import deque
from time import time
import gym


env = gym.make('LunarLander-v2')
obs = env.observation_space.shape[0]
action_space = env.action_space.n
agent = Agent(state_size=obs, action_size=action_space)
env.seed(0)
random.seed(0)

scores, avgs, times = [], [], []                   
eps, eps_decay = 1.0, 0.995       
start = time()   

for episode in range(1500):
    state = env.reset()
    score = 0

    # max steps per episodes
    for t in range(800):
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.experience(state, action, reward, next_state, done)
        state = next_state
        score += reward
        
        if done:
            break 

    # min value for epsilon is 0.01
    eps = max(0.01, eps_decay * eps)
    scores.append(score)
    avgs.append(np.mean(scores[-100:]))
    times.append(round((time()-start)/60, 2))           
    
    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores[-100:]):.2f}', end='')
    if episode % 100 == 0: print("")
    
    if avgs[-1] > 200:
        break

torch.save(agent.model, 'model.pth')

# plot the scores
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(np.arange(len(scores)), scores)
ax1.plot(np.arange(len(avgs)), avgs)
ax1.set_xlabel('Episode #')
ax1.set_ylabel('Score')

ax2.plot(np.arange(len(times)), times, 'g-')
ax2.set_ylabel('Time (min)')
plt.savefig('scores.png')
plt.show()
