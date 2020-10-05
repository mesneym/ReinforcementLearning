import torch.nn as nn 
import torch.optim as optim
import torch
from Network import *
from collections import deque
import random
import numpy as np
from replay_memory import ReplayBuffer



class DQN():
    def __init__(self,states,actions,alpha,gamma,epsilon,epsilon_min,epsilon_decay,replay_buffer_sz,batch,path,path_pred):
        self.Q = Network(states.shape,actions,alpha,path)
        self.Q_pred = Network(states.shape,actions,alpha,path_pred)

        # self.memory = deque(maxlen=replay_buffer_sz)
        self.memory = ReplayBuffer(replay_buffer_sz, states.shape, actions)
        self.batch = batch
        self.learn_cnt = 0

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = actions
        self.Q.path = path
        self.Q_pred.path = path_pred
        

    def e_greedy_policy(self,s):
        p = random.random()
        s = torch.tensor([s],dtype=torch.float).to(self.Q.device)
        # s = torch.unsqueeze(axis=0)
        a = torch.argmax(self.Q.forward(s)).item() if(p>self.epsilon) else np.random.randint(0,self.actions) 
        return a
        
    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch)

        states = torch.tensor(state).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        dones = torch.tensor(done).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        states_ = torch.tensor(new_state).to(self.Q.device)

        return states, actions, rewards, states_, dones 

    def store(self, s,a,r,ns,done):
        # self.memory.append([s,a,r,ns,done])
        self.memory.store_transition(s,a,r,ns,done)

    
    def update_target_network(self):
        self.Q_pred.load_state_dict(self.Q.state_dict())
       

    def save_models(self):
        self.Q.save_checkpoint(self.Q.path)
        self.Q_pred.save_checkpoint(self.Q_pred.path)
     

    def load_models(self):
        self.Q.load_checkpoint()
        self.Q_pred.load_checkpoint()

      
    def learn(self):
        if self.memory.mem_cntr < self.batch:
            return

        self.Q.optimizer.zero_grad()


        if(self.learn_cnt >= 1000):   #only update network after 1000 steps
            self.learn_cnt = 0
            self.update_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch)

        q_pred = self.Q.forward(states)[indices, actions]
        q_next = self.Q_pred.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.learn_cnt += 1

        self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)




            

            
            
            

             










    










    
    








