import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.a = nn.Conv2d(1,16,9)
        self.b = nn.Conv2d(16,16,7)
        self.l = nn.Linear(1024,150)
        self.l2 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.b(x)
        x = x.view(-1,1024)
        x = F.relu(x)
        x = self.l(x)
        x = F.relu(x)
        x = self.l2(x)
        return x