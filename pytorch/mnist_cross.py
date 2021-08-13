from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch, torchvision

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_data = datasets.MNIST('datasets', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_data = datasets.MNIST('datasets', train=False, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = relu(self.l3(x))
        x = relu(self.l4(x))
        x = self.l5(x)
        return x


model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
        
'''
运算结果：
[1,  300] loss: 2.190
[1,  600] loss: 0.912
[1,  900] loss: 0.416
Accuracy on test set: 89 %
[2,  300] loss: 0.319
[2,  600] loss: 0.272
[2,  900] loss: 0.224
Accuracy on test set: 93 %
[3,  300] loss: 0.191
[3,  600] loss: 0.170
[3,  900] loss: 0.155
Accuracy on test set: 95 %
[4,  300] loss: 0.128
[4,  600] loss: 0.126
[4,  900] loss: 0.120
Accuracy on test set: 96 %
[5,  300] loss: 0.101
[5,  600] loss: 0.094
[5,  900] loss: 0.093
Accuracy on test set: 97 %
[6,  300] loss: 0.077
[6,  600] loss: 0.077
[6,  900] loss: 0.074
Accuracy on test set: 97 %
[7,  300] loss: 0.065
[7,  600] loss: 0.058
[7,  900] loss: 0.064
Accuracy on test set: 97 %
[8,  300] loss: 0.050
[8,  600] loss: 0.050
[8,  900] loss: 0.052
Accuracy on test set: 97 %
[9,  300] loss: 0.040
[9,  600] loss: 0.041
[9,  900] loss: 0.041
Accuracy on test set: 97 %    
[10,  300] loss: 0.035
[10,  600] loss: 0.031
[10,  900] loss: 0.034
Accuracy on test set: 97 %
'''
