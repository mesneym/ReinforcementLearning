import numpy as np
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import random
import torch


class DQN():
    def __init__(self,states,actions,alpha,gamma,epsilon,epsilon_decay,epsilon_min):
        # self.Q = np.random.rand(states,actions) #Q learning is suceptible to initial values
        # self.Q[states-1,actions-1] = 0
        # self.Q = np.zeros((states,actions))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = Network(*states,actions,alpha).to(self.device)
        # self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = actions

    def learn(self,s,a,r,ns):
        self.Q.optimizer.zero_grad() 
        s_t = torch.tensor(s,dtype=torch.float).to(self.device)
        r_t = torch.tensor(r,dtype=torch.float).to(self.device)
        ns_t = torch.tensor(ns,dtype=torch.float).to(self.device)

        # self.Q[s,a] += self.alpha*(r + self.gamma*np.max(self.Q[ns,:]) - self.Q[s,a])  
        loss = self.Q.loss(r+ self.gamma*self.Q.forward(ns_t).max(), self.Q.forward(s_t)[a]).to(self.device)
        loss.backward()
        self.Q.optimizer.step()
        
        self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)

    
    def e_greedy_policy(self,s):
        p = random.random()
        s_t = torch.tensor(s,dtype=torch.float).to(self.device)
        a = torch.argmax(self.Q.forward(s_t)).item() if(p>self.epsilon) else np.random.randint(0,self.actions)
        return a


class Network(nn.Module):
    def __init__(self,s,a,alpha):
       super().__init__()
       self.relu = nn.ReLU()
       self.Fc1 = nn.Linear(s,128) 
       self.Fc2 = nn.Linear(128,a)

       self.optimizer = optim.Adam(self.parameters(),lr=alpha)
       self.loss = nn.MSELoss()

    def forward(self,x):
        x = self.relu(self.Fc1(x))
        x = self.Fc2(x)
        return x




