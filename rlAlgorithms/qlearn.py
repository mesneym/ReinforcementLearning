import random 
import numpy as np

class Qlearn:

    def __init__(self,states,actions,alpha,gamma,epsilon,epsilon_decay,epsilon_min):
        # self.Q = np.random.rand(states,actions) #Q learning is suceptible to initial values
        # self.Q[states-1,actions-1] = 0
        self.Q = np.zeros((states,actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = actions

    
    def learn(self,s,a,r,ns):
        self.Q[s,a] += self.alpha*(r + self.gamma*np.max(self.Q[ns,:]) - self.Q[s,a])  
        self.epsilon = max(self.epsilon_min,self.epsilon*self.epsilon_decay)

    
    def e_greedy_policy(self,s):
        p = random.random()
        a = np.argmax(self.Q[s,:]) if(p>self.epsilon) else np.random.randint(0,self.actions)
        return a
    















