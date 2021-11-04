import torch
import torch.nn as nn
import torchvision
import torch.optim as opt
from Net import Net

train_data = torchvision.datasets.QMNIST(
    './QMNIST/train',
    train=True,download=True,transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.QMNIST(
    './QMNIST/test',
    train=False,download=True,transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=4,shuffle=True)


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(10):
    runnning_loss = 0.0
    for i,(inputs,labels) in enumerate(train_loader,0):
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        runnning_loss += loss.item()

        print(f"epoch:{epoch}, loss:{runnning_loss}")
        runnning_loss = 0.0