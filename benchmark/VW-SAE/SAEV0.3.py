import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

"""
质量监督堆叠自编码器的代码
"""
# 数据集定义方式
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]

# 自编码器的定义
class AutoEncoder(nn.Module):
    def __init__(self, dim_X, dim_H):
        super(AutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        self.act = torch.sigmoid

        self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)

    def forward(self, X, rep=False):

        H = self.act(self.encoder(X))
        if rep is False:
            return self.act(self.decoder(H))
        else:
            return H

# 堆叠自编码器定义生成
class StackedAutoEncoder(nn.Module):
    def __init__(self, size, device=torch.device('cuda:0')):
        super(StackedAutoEncoder, self).__init__()
        self.AElength = len(size)
        self.SAE = []
        self.device = device

        for i in range(1, self.AElength):
            self.SAE.append(AutoEncoder(size[i-1], size[i]).to(device))

        self.proj = nn.Linear(size[self.AElength-1], 1)

    def forward(self, X, NoL, PreTrain=False):
        """
        :param X: 进口参数
        :param NoL: 第几层
        :param PreTrain: 是不是无监督预训练
        :return:
        """
        out = X
        if PreTrain is True:
            # SAE的预训练
            if NoL == 0:
                return out, self.SAE[NoL](out)

            else:
                for i in range(NoL):
                    # 第N层之前的参数给冻住
                    for param in self.SAE[i].parameters():
                        param.requires_grad = False

                    out = self.SAE[i](out, rep=True)
                # 训练第N层
                inputs = out
                out = self.SAE[NoL](out)
                return inputs, out
        else:
            for i in range(self.AElength-1):
                # 做微调
                for param in self.SAE[i].parameters():
                    param.requires_grad = True

                out = self.SAE[i](out, rep=True)
            out = torch.sigmoid(self.proj(out))
            return out

# 单层自编码器训练函数
def trainAE(model, trainloader, epochs, trainlayer, lr):

    optimizer = torch.optim.Adam(model.SAE[trainlayer].parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for j in range(epochs):
        sum_loss = 0
        for X, y in trainloader:
            Hidden, Hidden_reconst = model(X, trainlayer, PreTrain=True)
            loss = loss_func(Hidden, Hidden_reconst)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
        print('无监督预训练第{}层的第{}个epoch:'.format(trainlayer+1, j + 1), ',其Loss的大小是:{}'.format(loss.data.cpu().numpy()))

    return model

# SAE训练的代码模型
class SAEModel(BaseEstimator, RegressorMixin):
    def __init__(self, AEList=[13, 10, 7, 5], sup_epoch=300, unsp_epoch=200, unsp_batch_size=64, sp_batch_size=64, sp_lr=0.03,
                 unsp_lr=0.01, device=torch.device('cuda:0'), seed=1024):
        super(SAEModel, self).__init__()
        torch.manual_seed(seed)

        # 参数分配
        self.AEList = AEList
        self.num_AE = len(AEList) - 1
        self.unsp_epoch = unsp_epoch
        self.sup_epoch = sup_epoch
        self.unsp_batch_size = unsp_batch_size
        self.sp_batch_size = sp_batch_size
        self.unsp_lr = unsp_lr
        self.sp_lr = sp_lr
        self.device = device
        self.seed = seed

        # SAE模型的创建
        self.StackedAutoEncoderModel = StackedAutoEncoder(size=AEList, device=device).to(device)

        # 有多少AE就要单独定义多少次SAE
        self.optimizer = optim.Adam(
            [
                {'params': self.StackedAutoEncoderModel.parameters(), 'lr': self.unsp_lr},
                {'params': self.StackedAutoEncoderModel.SAE[0].parameters(), 'lr': self.sp_lr},
                {'params': self.StackedAutoEncoderModel.SAE[1].parameters(), 'lr': self.sp_lr},
                {'params': self.StackedAutoEncoderModel.SAE[2].parameters(), 'lr': self.sp_lr}
            ])

        self.loss_func = nn.MSELoss()

    # 数据拟合
    def fit(self, X, y):
        # 转换dataset
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.device),
                            torch.tensor(y, dtype=torch.float32, device=self.device),
                            '2D')

        un_trainloader = DataLoader(dataset, batch_size=self.unsp_batch_size, shuffle=True)
        trainloader = DataLoader(dataset, batch_size=self.sp_batch_size, shuffle=True)

        self.StackedAutoEncoderModel.train()

        for i in range(self.num_AE):
            print('自编码器训练第{}层:'.format(i + 1))
            self.StackedAutoEncoderModel = trainAE(model=self.StackedAutoEncoderModel, trainloader=un_trainloader,
                                                   epochs=self.unsp_epoch, trainlayer=i, lr=self.unsp_lr)
            print('自编码器第{}层训练完成!'.format(i + 1))

        Loss = []
        for i in range(self.sup_epoch):
            sum_loss = 0
            for batch_X, batch_y in trainloader:
                pre = self.StackedAutoEncoderModel(batch_X, i, PreTrain=False)
                loss = self.loss_func(pre, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().item()
            print('有监督微调第{}轮的Loss是{}'.format(i+1, loss.data.cpu().numpy()))
            Loss.append(sum_loss)
        # 绘制损失函数曲线
        plt.figure()
        plt.plot(range(len(Loss)), Loss, color='b')
        plt.show()

        return self

    # 预测数据
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.StackedAutoEncoderModel.eval()
        with torch.no_grad():
            y = self.StackedAutoEncoderModel(X, 0, PreTrain=False).cpu().numpy()

        return y

# 数据读取:
data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')
data = data.values

# 数据转化:
x_temp = data[:, :7]
y_temp = data[:, 7]


x_new = np.zeros([2390, 13])
x_6 = x_temp[:, 4]
x_9 = (x_temp[:, 5] + x_temp[:, 6])/2
x_new[:, :5] = x_temp[4: 2394, :5]

x_new[:, 5] = x_6[3: 2393]
x_new[:, 6] = x_6[2: 2392]
x_new[:, 7] = x_6[1: 2391]
x_new[:, 8] = x_9[4: 2394]


x_new[:, 9] = y_temp[3: 2393]
x_new[:, 10] = y_temp[2: 2392]
x_new[:, 11] = y_temp[1:2391]
x_new[:, 12] = y_temp[:2390]
y_new = y_temp[4: 2394]
y_new = y_new.reshape([-1, 1])

#划分数据集
# x_new = torch.from_numpy(x_new).float()
# y_new = torch.from_numpy(y_new).float()
train_x = x_new[:1000, :]
train_y = y_new[:1000]

x_validation = x_new[1000:1600, :]
y_validation = y_new[1000:1600]

test_x = x_new[1600:2390, :]
test_y = y_new[1600:2390]

# 开始搞活
mdl = SAEModel(AEList=[13, 10, 7, 5], sup_epoch=100, unsp_epoch=300, unsp_batch_size=50, sp_batch_size=20, sp_lr=0.03,
               unsp_lr=0.01, device=torch.device('cuda:0'), seed=1024).fit(train_x, train_y)

output_train = mdl.predict(train_x)
output_test = mdl.predict(test_x)


# 训练集画图
plt.figure()
plt.plot(range(len(output_train)), output_train, color='b', label='y_trainpre')
plt.plot(range(len(output_train)), train_y, color='r', label='y_true')
plt.legend()
plt.show()
train_rmse = np.sqrt(mean_squared_error(output_train, train_y))
train_r2 = r2_score(output_train, train_y)
print('train_rmse = ' + str(round(train_rmse, 5)))
print('r2 = ', str(train_r2))

# 测试集画图
plt.figure()
plt.plot(range(len(output_test)), output_test, color='b', label='y_testpre')
plt.plot(range(len(output_test)), test_y, color='r', label='y_true')
plt.legend()
plt.show()
test_rmse = np.sqrt(mean_squared_error(output_test, test_y))
test_r2 = r2_score(output_test, test_y)
print('test_rmse = ' + str(round(test_rmse, 5)))
print('r2 = ', str(test_r2))