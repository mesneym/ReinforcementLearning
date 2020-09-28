import gym
import matplotlib.pyplot as plt
import numpy as np



env = gym.make("FrozenLake-v0")
# env = gym.make("FrozenLake-v0",is_slippery=False)
env.reset()

results = []
rewards = []

#l = 0 d = 1 r = 2 u = 3
#SFFF
#FHFH
#FFFH
#HFFG


policy = {0:2,1:2,2:1,3:0,4:1,6:1,8:2,9:1,10:1,13:2,14:2}
for i in range(1000):
    done = False
    state = env.reset()
    value = 0

    while not done:
        # env.render()
        action = policy[state]
        state,reward,done,info = env.step(action) 
        value += reward
        # print(f"reward is {reward:.2f}")
    rewards.append(value)

    if(i%10==0):
        results.append(np.mean(rewards[-10:]))


x = np.arange(len(results))
plt.plot(x,results)
plt.show()
env.close()




