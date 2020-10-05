import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self,s,a,alpha,path):
        super().__init__()
        
        self.path = path
        
        self.conv1 = nn.Conv2d(in_channels=s[0], out_channels=32, kernel_size=(8,8), stride =4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=(4,4), stride =2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=(3,3), stride =1)
        self.Fc1 = nn.Linear(self.__find_dim(s),512)
        self.Fc2 = nn.Linear(512,a)

        self.relu = nn.ReLU()
        self.optimizer = optim.RMSprop(self.parameters(),lr=alpha)
        self.loss = nn.MSELoss()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
        self.to(self.device)
    
    def __find_dim(self,s):
        x = torch.rand((1,*s))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.prod(torch.tensor(x.shape)).item()

    def save_checkpoint(self,path):
        print("saving checkpoint")
        torch.save(self.state_dict(),self.path)

    def load_checkpoint(self):
        print("loading checkpoint")
        self.load_state_dict(torch.load(self.path))

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape((x.shape[0],-1))
        x = self.relu(self.Fc1(x))
        x = self.Fc2(x)
        return x


if __name__ == '__main__':
   s = torch.rand((3,82,82))
   a = 10
   model = Network(s.shape,a,0.01,'ad')
   print(model)

   s_ex = torch.rand((10,3,82,82))
   print(model(s_ex).shape)
  







