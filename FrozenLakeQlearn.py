import gym 
import matplotlib.pyplot as plt
from rlAlgorithms.qlearn import *


# Using Qlearning  

env = gym.make("FrozenLake-v0")
QAgent = Qlearn(states = 16,
                actions = 4,
                alpha = 0.001,
                gamma = 0.9,
                epsilon = 1,
                epsilon_decay = 0.9999995,
                epsilon_min = 0.01)


rewards = []
results = []

for i in range(500000):
    success = False
    s = env.reset()
    score = 0

    while not success:
        # env.render()
        a = QAgent.e_greedy_policy(s)
        ns,r,success,info = env.step(a)
        QAgent.learn(s,a,r,ns) 
        s = ns

        score += r
  

    rewards.append(score)
    if(i%100==0):
        results.append(np.mean(rewards[-100:]))
        if i % 1000 == 0:
            print(f"episode num: {i}, win-pct: {results[-1]} epsilon : {QAgent.epsilon}")

env.close()

plt.plot(results)
plt.show()




