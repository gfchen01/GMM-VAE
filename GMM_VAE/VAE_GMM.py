import sys

import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes, load_digits
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


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


class VAE_GMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, category, output_dim):
        super(VAE_GMM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.category = category
        self.output_dim = output_dim

        # 编码器
        self.encoder_fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)

        # 整形编码
        self.encoder_z_mu = nn.Linear(in_features=hidden_dim, out_features=z_dim * category, bias=True)
        self.encoder_z_logvar = nn.Linear(in_features=hidden_dim, out_features=z_dim * category, bias=True)

        # 类别编码
        self.encoder_category = nn.Linear(in_features=hidden_dim, out_features=category, bias=True)

        # 解码器
        self.decoder_fc1 = nn.Linear(in_features=z_dim * category, out_features=hidden_dim * category, bias=True)
        self.decoder_fc2 = nn.Linear(in_features=hidden_dim * category, out_features=output_dim * category, bias=True)

        self.decoder_fc3 = nn.Linear(in_features=output_dim, out_features=output_dim, bias=True)

        # 激活函数
        # self.act = torch.relu
        self.act_gaussian = torch.sigmoid
        self.act_category = torch.sigmoid

    def sample_gumbel(self, shape, device, eps=1.0e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(shape=logits.size(), device=device)
        return torch.nn.functional.softmax(y / temperature, dim=-1)

    def gumbel_max_sample(self, logits, device):
        y = logits + self.sample_gumbel(shape=logits.size(), device=device)
        return y

    def gumbel_softmax_reparm(self, logits, temperature, device):


        if temperature == 0:
            y = self.gumbel_max_sample(logits=logits, device=device)
        else:
            y = self.gumbel_softmax_sample(logits=logits, temperature=temperature, device=device)

        # print('打印形状')
        # print(y.shape)
        # print(torch.sum(y, dim=-1))

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)

        y_hard = (y_hard - y).detach() + y
        return y_hard


    def forward(self, X, temperature):

        encoded = self.encoder_fc1(X)

        # log_category映射
        encoded_category = self.act_category(encoded)
        # encoded_category = encoded_category
        origin_log_category = self.encoder_category(encoded_category)
        # 归一化 category的形状是[batch_size, category, 1]
        category_real = torch.exp(origin_log_category)
        category_real = torch.softmax(input=category_real, dim=-1)
        # log分布
        log_category = torch.log(category_real)
        # gumbel-softmax重参化
        mask_matrix = self.gumbel_softmax_reparm(logits=log_category, temperature=temperature, device=X.device).unsqueeze(2)



        encoded_gaussian = self.act_gaussian(encoded)

        mu = self.encoder_z_mu(encoded_gaussian).view(-1, self.z_dim, self.category)
        logvar = self.encoder_z_logvar(encoded_gaussian).view(-1, self.z_dim, self.category)



        # gaussian 重参化
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        z = z.view(-1, self.z_dim * self.category)

        latent_feature = self.decoder_fc1(z)
        latent_feature = self.act_gaussian(latent_feature)
        latent_feature = self.decoder_fc2(latent_feature).view(-1, self.output_dim, self.category)
        latent_feature = self.act_gaussian(latent_feature)
        latent_feature = torch.bmm(input=latent_feature, mat2=mask_matrix).squeeze(2)
        output = self.decoder_fc3(latent_feature)
        return output, latent_feature, z, category_real, mu, logvar, mask_matrix


class VAE_GMMModel(BaseEstimator, RegressorMixin):
    # input_dim, hidden_dim, z_dim, category, output_dim
    def __init__(self, dim_X, dim_H, dim_Z, dim_y, dim_C, batch_size, lr, n_epoch, device, seed):
        super(VAE_GMMModel, self).__init__()

        # Set Seed
        torch.manual_seed(seed)

        self.dim_X = dim_X
        self.dim_H = dim_H
        self.dim_Z = dim_Z
        self.dim_y = dim_y
        self.dim_C = dim_C
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed
        self.n_epoch = n_epoch

        # Scaler
        # self.scaler_X = MinMaxScaler()
        # self.scaler_y = MinMaxScaler()

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # model
        self.loss_hist = []
        self.mse_hist = []
        self.Gaussian_KL_hist = []
        self.Category_KL_hist = []

        self.model = VAE_GMM(input_dim=dim_X, hidden_dim=dim_H, z_dim=dim_Z, category=dim_C, output_dim=dim_X+dim_y).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def loss_calculation(self, Posterior_Category, Prior_Category, Posterior_Gaussian_mu, Posterior_Gaussian_logvar, y_pred, y_real):
        batch_size = y_real.shape[0]
        # Gaussian的KL散度计算
        OneTensor_Gaussian = torch.ones_like(input=Posterior_Gaussian_mu)
        DKL_Gaussian = -0.5 * (OneTensor_Gaussian + Posterior_Gaussian_logvar - Posterior_Gaussian_mu * Posterior_Gaussian_mu - torch.exp(Posterior_Gaussian_logvar))
        DKL_Gaussian = torch.sum(input=DKL_Gaussian) / (batch_size * self.dim_C)

        # Category的KL散度计算
        Prior_Category = Prior_Category.repeat(batch_size, 1)

        DKL_Category = Posterior_Category * torch.log(Posterior_Category / Prior_Category)
        DKL_Category = torch.sum(DKL_Category) / (batch_size)

        # 损失函数计算
        pred_error = torch.nn.functional.mse_loss(input=y_pred, target=y_real, reduction='sum')/ (batch_size)

        return DKL_Gaussian, DKL_Category, pred_error

    def fit(self, X, y, PriorCategory=None, FineTuning=True):
        if PriorCategory is None:
            PriorCategory = (1/self.dim_C) * torch.ones([1, self.dim_C], device=self.device)



        y = y.reshape(-1, self.dim_y)

        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

        data_X = torch.tensor(data=X, dtype=torch.float32, device=self.device)
        data_y = torch.tensor(data=y, dtype=torch.float32, device=self.device)

        dataset = MyDataset(data=data_X, label=data_y, mode='2D')

        self.model.train()
        for i in range(self.n_epoch):

            # 温度计算
            temperature = math.exp(-0.025*(i)) * 20.0

            self.loss_hist.append(0)
            self.mse_hist.append(0)
            self.Gaussian_KL_hist.append(0)
            self.Category_KL_hist.append(0)
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                output, z_feature, z, category_real, mu, logvar, mask_matrix = self.model(X=batch_X, temperature=temperature)
                #
                # print('打印形状')
                # print(batch_X.shape)
                # print(batch_y.shape)
                reconst_real = torch.cat((batch_X, batch_y), dim=-1)


                DKL_Gaussian, DKL_Category, pred_error = self.loss_calculation(Posterior_Category=category_real, Prior_Category=PriorCategory, Posterior_Gaussian_mu=mu, Posterior_Gaussian_logvar=logvar, y_real=reconst_real, y_pred=output)

                loss = DKL_Gaussian + DKL_Category + pred_error
                # loss = DKL_Gaussian + DKL_Category + pred_error
                self.loss_hist[-1] += loss.item()
                self.mse_hist[-1] += pred_error.item()
                self.Gaussian_KL_hist[-1] += DKL_Gaussian.item()
                self.Category_KL_hist[-1] += DKL_Category.item()

                loss.backward()


                # for name, parms in self.model.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad)

                self.optimizer.step()

            print('Pretraining Epoch:{}, Loss:{}, MSE:{}, KL Gaussian:{}, KL Category:{}'.format(i + 1,
                                                                                                 str(round(self.loss_hist[-1], 6)),
                                                                                                 str(round(self.mse_hist[-1], 6)),
                                                                                                 str(round(self.Gaussian_KL_hist[-1], 6)),
                                                                                                 str(round(self.Category_KL_hist[-1], 6))
                                                                                                 ))

        print('Pretraining Optimization Finished')

        if FineTuning:
            for i in range(int(self.n_epoch*0.80)):

                # 温度计算
                temperature = 0.1

                self.loss_hist.append(0)
                self.mse_hist.append(0)
                self.Gaussian_KL_hist.append(0)
                self.Category_KL_hist.append(0)
                data_loader = DataLoader(dataset=dataset, batch_size=int(self.batch_size/2), shuffle=True)
                for batch_X, batch_y in data_loader:
                    self.optimizer.zero_grad()
                    output, z_feature, z, category_real, mu, logvar, mask_matrix = self.model(X=batch_X, temperature=temperature)
                    #
                    # print('打印形状')
                    # print(batch_X.shape)
                    # print(batch_y.shape)
                    reconst_real = batch_y
                    pred_y = output[:, -self.dim_y].view(-1, self.dim_y)
                    # print(reconst_real.shape)
                    # print(pred_y.shape)


                    DKL_Gaussian, DKL_Category, pred_error = self.loss_calculation(Posterior_Category=category_real, Prior_Category=PriorCategory, Posterior_Gaussian_mu=mu, Posterior_Gaussian_logvar=logvar, y_real=reconst_real, y_pred=pred_y)

                    loss = pred_error
                    # loss = DKL_Gaussian + DKL_Category + pred_error
                    self.loss_hist[-1] += loss.item()
                    self.mse_hist[-1] += pred_error.item()
                    self.Gaussian_KL_hist[-1] += DKL_Gaussian.item()
                    self.Category_KL_hist[-1] += DKL_Category.item()

                    loss.backward()


                    # for name, parms in self.model.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                    #           ' -->grad_value:', parms.grad)

                    self.optimizer.step()

                print('Epoch:{}, Loss:{}, kl_gaussian:{}, kl_cat{}'.format(i + 1, str(round(self.loss_hist[-1], 6)),
                                                                         str(round(self.Gaussian_KL_hist[-1], 6)),
                                                                         str(round(self.Category_KL_hist[-1], 6))))
            print('Fine-tuning Optimization Finished!')

        return self

    def predict(self, X):
        process_X = self.scaler_X.transform(X)
        test_X = torch.tensor(data=process_X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output, z_feature, z, category_real, mu, logvar, mask_matrix = self.model(X=test_X, temperature=0)
            mask_matrix = mask_matrix.squeeze(2)
            output = output[:, -1]
            output = output.view(-1, 1)

            output = output.cpu().numpy()
            output = self.scaler_y.inverse_transform(output)

            mask_matrix = mask_matrix.cpu().numpy()


        return output, mask_matrix





if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')
    # print(data)
    EPOCH = 300

    data = data.values

    # 数据转化:
    x_temp = data[:, :7]
    y_temp = data[:, 7]

    x_new = np.zeros([2390, 13])
    x_6 = x_temp[:, 4]
    x_9 = (x_temp[:, 5] + x_temp[:, 6]) / 2
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

    # 划分数据集
    # x_new = torch.from_numpy(x_new).float()
    # y_new = torch.from_numpy(y_new).float()
    train_x = x_new[:1600, :]
    train_y = y_new[:1600]

    x_validation = x_new[1000:1600, :]
    y_validation = y_new[1000:1600]

    test_x = x_new[1600:2390, :]
    test_y = y_new[1600:2390]

    # print('形状')
    # print(train_x.shape[1])
    # dim_X, dim_H, dim_Z, dim_y, dim_C, batch_size, lr, n_epoch, device, seed):
    mdl = VAE_GMMModel(dim_X=train_x.shape[1],
                       dim_H=13,
                       dim_Z=8,
                       dim_y=1,
                       dim_C=4,
                       batch_size=64,
                       lr=0.0010,
                       n_epoch=EPOCH,
                       device=DEVICE,
                       seed=1024)

    mdl = mdl.fit(X=train_x, y=train_y, PriorCategory=None, FineTuning=True)

    # torch.save(mdl, 'TrainedNet2')

    # sys.exit(0)

    train_y_predict, mask_matrix_train = mdl.predict(train_x)
    test_y_predict, mask_matrix_test = mdl.predict(test_x)

    # whole_y_predict, mask_matrix_whole = mdl.predict(x_temp)

    train_rmse = np.sqrt(mean_squared_error(train_y_predict, train_y))
    train_r2 = r2_score(train_y_predict, train_y)
    print('train_rmse = ' + str(round(train_rmse, 5)))
    print('r2 = ', str(train_r2))

    test_rmse = np.sqrt(mean_squared_error(test_y_predict, test_y))
    test_r2 = r2_score(test_y_predict, test_y)
    print('test_rmse = ' + str(round(test_rmse, 5)))
    print('r2 = ', str(test_r2))

    df = pd.DataFrame(mask_matrix_test)
    df.to_csv('mask.csv', sep=',')
    # ax = sns.heatmap(data=df, cmap='gray_r', square=True, linewidths=0.05, linecolor='black', cbar=True)
    # sns.despine(top=False, right=False, left=False, bottom=False)
    # label_y = ax.get_yticklabels()
    # plt.setp(label_y, rotation=360, horizontalalignment='right')
    # plt.setp(ax.patches, linewidth=3)
    # plt.show()

    plt.figure()
    plt.plot(range(len(test_y)), test_y_predict, color='b', label='y_testpre')
    plt.plot(range(len(test_y)), test_y, color='r', label='y_true')
    plt.legend()
    plt.show()
    # test_rmse = np.sqrt(mean_squared_error(output_test, test_y))
    # test_r2 = r2_score(output_test, test_y)
    # print('test_rmse = ' + str(round(test_rmse, 5)))
    # print('r2 = ', str(test_r2))