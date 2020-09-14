import gym
import matplotlib.pyplot as plt
import numpy as np



env = gym.make("FrozenLake-v0")
env.reset()

results = []
rewards = []


for i in range(1000):
    done = False
    env.reset()
    value = 0

    while not done:
        # env.render()
        newstate,reward,done,info = env.step(env.action_space.sample()) 
        value += reward
        # print(f"reward is {reward:.2f}")
    rewards.append(value)
        
    if(i%10==0):
        results.append(np.mean(rewards[-10:]))


x = np.arange(len(results))
plt.plot(x,results)
plt.show()
env.close()




