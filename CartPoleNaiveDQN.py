import gym
import matplotlib.pyplot as plt
from rlAlgorithms.dqn import *

env = gym.make("CartPole-v1")
state = env.observation_space.shape
actions = env.action_space.n

DQNAgent = DQN(states = state,
               actions = actions,
               alpha = 0.0001,
               gamma = 0.99,
               epsilon = 1,
               # epsilon_decay = 0.999985,
               epsilon_decay = 1e-5,
               epsilon_min = 0.01)

results = []
rewards = []

for i in range(10000):
    score = 0
    success = False
    s = env.reset()

    while not success:
        # env.render()
        a = DQNAgent.e_greedy_policy(s)
        ns,r,success,info = env.step(a)
        score += r
        DQNAgent.learn(s,a,r,ns)
        s = ns
    rewards.append(score) 

    if(i%100 ==0):
        results.append(np.mean(rewards[-100:]))
        print(f"episode num: {i}, win-pct: {results[-1]} epsilon : {DQNAgent.epsilon}")


plt.plot(results)
env.close()



