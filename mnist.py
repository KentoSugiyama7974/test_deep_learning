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

#train
for epoch in range(1):
    runnning_loss = 0.0
    for i,(inputs,labels) in enumerate(train_loader,0):
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        runnning_loss += loss.item()

        if i%100 == 99:
            print(f"epoch:{epoch},{i} loss:{runnning_loss}")
            runnning_loss = 0.0

#test
correct = 0
total = 0

with torch.no_grad():
    for (images,labels) in test_loader:
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"accuracy:{float(correct/total)}")