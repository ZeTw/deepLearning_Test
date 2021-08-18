from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
import torch, torchvision

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()


# c表示颜色数据 X 第0行表示该点的横坐标 第一行表示该点的纵坐标  Y表示该点的颜色
# s为每个点的面积 cmap 指的是colormap
# 根据上述信息，要做的是一个二分类的神经网络，即将不同颜色的点区分开
# 每个样本对应的输入有两个横纵坐标，对应的输出有一个Y
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

plt.show()

class LogisticNet(torch.nn.Module):
    def __init__(self, nodes):
        super(LogisticNet, self).__init__()
        # 单隐层的神经网络
        self.linear = torch.nn.Linear(2, nodes, bias=True)
        self.linear2 = torch.nn.Linear(nodes, 1, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.linear(x))
        x = self.sigmoid(self.linear2(x))
        return x


class MyDataset(Dataset):
    def __init__(self):
        X, Y = load_planar_dataset()
        self.len = X.shape[1]
        self.x_data = torch.tensor(X.T).to(torch.float32)
        self.y_data = torch.tensor(Y.T).to(torch.float32)
        

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]




def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(train_loader, 0):
            outputs = model(inputs)
            predicted = outputs.gt(0.5).to(torch.int)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


def plot_decision_boundary():  # 绘制决策边界

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1  # 横坐标的最大值 和最小值

    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1  # 纵坐标的最大值和最小值
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # 间隔0.01形成矩阵 然后生成网格点坐标矩阵
    # 由于横坐标最大最小值之差和纵坐标最大最小值之差 不一定一致，因此按照固定间隔生成的横坐标矩阵和纵坐标矩阵的维度也不一定一致
    # 而使用 meshgrid函数 能够保证xx 和 yy矩阵的维度是一致的 一个横坐标对应一个纵坐标
    # 将原有的矩阵在行方向上进行扩展 组成新的矩阵
    inputs = np.c_[xx.ravel(), yy.ravel()]
    Z = torch.tensor(inputs).to(torch.float32)
    outputs = model(Z).gt(0.5).to(torch.int).reshape(xx.shape)  # 将预测结果 reshape为 xx矩阵维度

    # 然后画出图
    plt.contourf(xx, yy, outputs, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()


model = LogisticNet(10)

dataset = MyDataset()
train_loader = DataLoader(dataset=dataset, batch_size=400, shuffle=True)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

if __name__ == '__main__':
    for epoch in range(5000):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            if epoch % 1000 == 0:
                print(epoch, loss.item())

            loss.backward()

            optimizer.step()
        if epoch % 1000 == 0:
            test()

plot_decision_boundary()
