import time

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
print(torch.cuda.is_available())
# 读取数据
df1 = pd.read_csv('input.csv', header=None)
df2 = pd.read_csv('output.csv', header=None)

# 设置随机种子
np.random.seed(42)

# 划分数据集
X = df1.iloc[1:1000001, :41] #
Y = df2.iloc[2:1000002, :]
print(Y)
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2,
random_state=42)
print(train_features.shape)
protion = 0.00125
num = train_features.shape[0]
train_features = train_features[:int(protion * num)]

num2 = train_labels.shape[0]
train_labels = train_labels[:int(protion * num2)]
print(train_labels.shape)
# print(train_features, test_features, train_labels, test_labels)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(41, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# 实例化模型
net = Net().to('cuda:0')

# 定义损失函数和优化器
criterion = nn.BCELoss().to('cuda:0')
optimizer = optim.Adam(net.parameters(), lr=0.001)  # learning rate ,优化器
s = time.time()
epoc = 3000
# 训练模型
for epoch in range(epoc):
    inputs = torch.tensor(train_features.values.astype(np.float64), dtype=torch.float32).to('cuda:0')
    # print(train_labels.values, type(train_labels.values))
    labels = torch.tensor(train_labels.values.astype(np.float64), dtype=torch.float32).to('cuda:0')

    # 将梯度清零
    optimizer.zero_grad()

    # 前向传播
    outputs = net(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()

    # 更新权重
    optimizer.step()

    # 打印训练过程
    if (epoch + 1) % 100 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epoc, loss.item()), '100 Epoch time: ', time.time()-s)
        s = time.time()

# 测试模型
with torch.no_grad():
    inputs = torch.tensor(test_features.values.astype(np.float64), dtype=torch.float32).to('cuda:0')
    labels = torch.tensor(test_labels.values.astype(np.float64), dtype=torch.float32).to('cuda:0')
    outputs = net(inputs)
    predicted = (outputs >= 0.5).float().to('cuda:0')
    accuracy = (predicted == labels).sum().item() / len(labels)
    print('Accuracy: %.2f%%' % (accuracy * 100))