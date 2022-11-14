import pandas as pd
import math
import random
import numpy as np
import torch
import torch.nn as nn
import scipy
import sys
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# np.random.seed(seed=1024)
# torch.manual_seed(seed=1024)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1024)


class DPMM(nn.Module):
    def __init__(self, variationalT, alphaDP, data, device, inputDim, hiddenDim=20):
        super(DPMM, self).__init__()
        self.variationalT = variationalT
        self.clusterNum = variationalT
        self.alphaDP = alphaDP
        self.device = device
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim


        # 数据的继承和修改
        self.dataNumber = data.shape[0]
        self.dataDimension = data.shape[1]
        self.data = torch.tensor(data=data, dtype=torch.float32, device=device)
        self.inputData = self.data[:, 0:inputDim].view(self.dataNumber, -1)
        self.outputData = self.data[:, -1].view(self.dataNumber, -1)


        # Beta分布先验超参
        self.priorBetaGamma1 = torch.ones(1, variationalT, dtype=torch.float32, device=device)
        self.priorBetaGamma2 = torch.ones(1, variationalT, dtype=torch.float32, device=device)

        # 变分分布中Beta分布的参数
        self.variationalBetaGamma1 = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        self.variationalBetaGamma2 = torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # 变分分布中的Categorical分布参数
        self.npPhi = np.random.multinomial(n=1, pvals=(1.0 / float(variationalT) * np.ones(variationalT)), size=self.dataNumber)
        self.variationalPhi = torch.tensor(data=self.npPhi, dtype=torch.float32, device=device)

        # 进行回归用的参数
        # 权重参数
        self.variationalWM = 0.0 * torch.ones([variationalT, (hiddenDim + 1)], dtype=torch.float32, device=device)
        self.variationalWS = 1.0e0 * torch.ones([(hiddenDim + 1), (hiddenDim + 1), variationalT], dtype=torch.float32, device=device)

        # 权重的方差
        self.variationalZetaA = 1.0e-4 * torch.ones([1, variationalT], dtype=torch.float32, device=device)
        self.variationalZetaB = 1.0e0 * torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # 残差的分布超参数
        self.variationalEpsilonA = 1.0 * torch.ones([1, variationalT], dtype=torch.float32, device=device)
        self.variationalEpsilonB = 1.0e-3 * torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # T个高斯成分 * 数据维度
        # 均值的变分超参数
        self.variationalGaussianMeanMu = torch.ones([inputDim, variationalT], dtype=torch.float32, device=device)
        self.variationalGaussianVarKappa = torch.ones([1, variationalT], dtype=torch.float32, device=device)
        # 方差的变分超参数
        self.variationalWishartPsi = torch.ones([inputDim, inputDim, variationalT], dtype=torch.float32, device=device)
        self.variationalWishartNu = torch.ones([1, variationalT], dtype=torch.float32, device=device)

        # 均值的先验超参数
        # self.priorGaussianMeanMu = torch.mean(input=self.inputData, dim=0).unsqueeze(1)
        self.priorGaussianMeanMu = torch.zeros(size=(inputDim, 1), device=device)
        self.priorGaussianVarKappa = 1000.0 * torch.ones([1], dtype=torch.float32, device=device)
        # 方差的先验超参数
        self.priorWishartPsi = 500.0 * torch.eye(inputDim, dtype=torch.float32, device=device)
        # self.priorWishartPsi = 1.0 * torch.ones((inputDim, inputDim), dtype=torch.float32, device=device)
        # self.priorWishartNu = float(self.dataDimension + 1) * torch.ones([1], dtype=torch.float32, device=device)
        self.priorWishartNu = 100.0 * torch.ones([1], dtype=torch.float32, device=device)

        self.priorZetaA = self.variationalZetaA
        self.priorZetaB = self.variationalZetaB

        self.priorEpsilonA = self.variationalEpsilonA
        self.priorEpsilonB = self.variationalEpsilonB

        # self.weightELM = (1 - (-1)) * torch.rand((inputDim), (inputDim)) + (-1)
        # self.weightELM = torch.randn((inputDim), (inputDim))
        # self.weightELM = (1 - (-1)) * torch.rand((inputDim), (inputDim)) + (-1)
        weightELMTemp = np.random.rand(inputDim, hiddenDim)
        # self.weightELM = (1 - (-1)) * torch.rand((inputDim, hiddenDim), device=device) + (-1)
        self.weightELM = (1 - (-1)) * torch.tensor(data=weightELMTemp, dtype=torch.float32, device=device) + (-1)
        # self.biasELM = (1 - (-1)) * torch.rand(inputDim, 1) + (-1)
        biasELMTemp = np.random.randn(hiddenDim, 1)
        # self.biasELM = torch.randn((hiddenDim, 1), device=device)
        self.biasELM = torch.tensor(data=biasELMTemp, dtype=torch.float32, device=device)

    def mapFunction(self, inputData):

        outputData = inputData @ self.weightELM + self.biasELM.T.repeat(inputData.shape[0], 1)
        outputData = torch.sigmoid(outputData)

        return outputData

    # def vbMStep(self, BetaGamma1, BetaGamma2, Phi, GaussianMuMk, GaussianMuKappa, WishartPsi, WishartNu):
    def vbMStep(self, BetaGamma1, Phi, inputData, outputData):


        # Gamma1参数计算 维度: (1 * t)
        BetaGamma1 = torch.ones_like(input=BetaGamma1, device=self.device) + torch.sum(input=Phi, dim=0, keepdim=True)

        # 计算Phi及其掩码矩阵, 避免后续for循环
        tempGamma2Phi = Phi.unsqueeze(0).repeat(self.variationalT, 1, 1)

        # torch.tril(input=torch.ones([4, 4]), diagonal=0).unsqueeze(2).repeat(1, 1, 4).permute(0, 2, 1)
        # torch.flip(input=u, dims=[0, 2])
        # torch.flip(input=torch.tril(input=torch.ones([self.variationalT, self.variationalT]), diagonal=0).unsqueeze(2).repeat(1, 1, self.dataNumber).permute(0, 2, 1), dims=[0, 2])

        # tempGamma2Mask = torch.flip(input=torch.tril(input=torch.ones([self.variationalT, self.variationalT], device=self.device, dtype=torch.float32), diagonal=0).unsqueeze(2).repeat(1, 1, self.dataNumber).permute(0, 2, 1), dims=[0, 2])

        tempGamma2Mask = torch.flip(input=torch.tril(input=torch.ones([self.variationalT, self.variationalT], device=self.device), diagonal=-1).unsqueeze(2).repeat(1, 1, self.dataNumber).permute(0, 2, 1), dims=[0, 2])

        # Gamma2参数计算 维度: (1 * t)
        BetaGamma2 = (float(self.alphaDP) + torch.sum(input=(tempGamma2Phi * tempGamma2Mask), dim=[1, 2], keepdim=False).T).unsqueeze(0)

        # 指数分布族参数计算
        # 预先分配空内存
        tempExpoentialTaoT11 = torch.zeros([self.inputDim, self.inputDim, self.variationalT], device=self.device, dtype=torch.float32)



        # 第一个自然参数的计算
        for i in range(self.variationalT):
            # 一共有t个成分, 打循环:
            temp1ForExpoentialTaoT11 = -0.5 * (self.priorWishartPsi + self.priorGaussianVarKappa * (self.priorGaussianMeanMu) @ (self.priorGaussianMeanMu.T)).view(-1, 1)
            # print('打印形态')
            # print(self.inputData.shape)
            temp2ForExpoentialTaoT11 = -0.5 * ((torch.bmm(input=inputData.unsqueeze(2), mat2=inputData.unsqueeze(1)).view(self.dataNumber, self.inputDim * self.inputDim)) * (Phi[:, i].unsqueeze(1).repeat(1, self.inputDim * self.inputDim)))
            tempExpoentialTaoT11[:, :, i] = (temp1ForExpoentialTaoT11.squeeze(1) + torch.sum(temp2ForExpoentialTaoT11, dim=0, keepdim=False)).view(self.inputDim, self.inputDim)



        # 第二个自然参数的计算 维度: (d * t)
        tempExpoentialTaoT12 = self.priorGaussianVarKappa * (self.priorGaussianMeanMu.repeat(1, self.variationalT)) + inputData.T @ Phi

        # 第三个自然参数的计算 维度: (1 * t)
        tempExpoentialTaoT21 = -0.5 * torch.sum(Phi, dim=0, keepdim=True) - 0.5 * (self.priorWishartNu + self.inputDim + 2)

        # 第四个自然参数的计算 维度: (1 * t)
        tempExpoentialTaoT22 = self.priorGaussianVarKappa.unsqueeze(1).repeat(1, self.variationalT) + torch.sum(Phi, dim=0, keepdim=True)


        # 后验分布参数
        # 变分高斯分布
        variationalGaussianVarKappa = tempExpoentialTaoT22
        variationalGaussianMeanMu = tempExpoentialTaoT12 / tempExpoentialTaoT22.repeat(self.inputDim, 1)

        # print('打印设备')
        # print(self.priorGaussianVarKappa.device)

        # 变分威沙特分布
        variationalWishartNu = -2.0 * (tempExpoentialTaoT21) - 2.0 - float(self.inputDim)


        variationalWishartPsi = -2.0 * tempExpoentialTaoT11 - variationalGaussianVarKappa.unsqueeze(0).repeat(self.inputDim, self.inputDim, 1) * torch.bmm(input=variationalGaussianMeanMu.unsqueeze(1).permute(2, 0, 1), mat2=variationalGaussianMeanMu.unsqueeze(1).permute(2, 1, 0)).permute(2, 1, 0)

        # 对线性回归系数进行计算
        # 计算多模态条件下的回归系数
        inputDataMap = self.mapFunction(inputData=inputData)
        xTil = torch.cat((inputDataMap, torch.ones((self.dataNumber, 1), device=self.device)), dim=1).unsqueeze(0).repeat(self.variationalT, 1, 1)
        epsilonExp = self.variationalEpsilonA / self.variationalEpsilonB
        # 1 * variationalT
        zetaExp = self.variationalZetaA / self.variationalZetaB

        # 计算 T * n * n
        variationalWSTerm1 = torch.diag_embed(input=Phi.permute(1, 0), offset=0, dim1=0).permute(1, 0, 2)
        # T * (d + 1) * (d + 1)
        variationalWSTerm1 = torch.bmm(input=xTil.permute(0, 2, 1), mat2=variationalWSTerm1)
        variationalWSTerm1 = torch.bmm(input=variationalWSTerm1, mat2=xTil)


        variationalWSTerm1 = epsilonExp.permute(1, 0).unsqueeze(1).repeat(1, (self.hiddenDim + 1), (self.hiddenDim + 1)) * variationalWSTerm1

        variationalWSTerm2 = torch.eye((self.hiddenDim + 1), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.variationalT, 1, 1)



        variationalWSTerm2 = zetaExp.permute(1, 0).unsqueeze(1).repeat(1, (self.hiddenDim + 1), (self.hiddenDim + 1)) * variationalWSTerm2

        # T * (d + 1) * (d + 1)
        variationalWS = torch.inverse(variationalWSTerm1 + variationalWSTerm2)

        # print(variationalWS.shape)
        # print(xTil.shape)

        variationalWMTemp = torch.bmm(input=variationalWS, mat2=xTil.permute(0, 2, 1))
        variationalWMTemp = torch.bmm(input=variationalWMTemp, mat2=torch.diag_embed(input=Phi.permute(1, 0), offset=0, dim1=0).permute(1, 0, 2))
        variationalWMTemp = torch.bmm(input=variationalWMTemp, mat2=outputData.unsqueeze(0).repeat(self.variationalT, 1, 1))

        # print(outputData.shape)

        # T * (d + 1) * 1
        variationalWM = zetaExp.permute(1, 0).unsqueeze(1).repeat(1, (self.hiddenDim + 1), 1) * variationalWMTemp

        variationalZetaA = self.priorZetaA + (self.hiddenDim + 1) * 0.5

        # print('打印形态')
        # print(variationalWM.shape)
        # temp1 = torch.bmm(input=variationalWM.permute(0, 2, 1), mat2=variationalWM)



        variationalZetaB = self.priorZetaB + (torch.bmm(input=variationalWM.permute(0, 2, 1), mat2=variationalWM).squeeze(2)
                                              + torch.diagonal(variationalWS, dim1=-2, dim2=-1).sum(-1).unsqueeze(1)).permute(1, 0) * 0.5


        variationalEpsilonA = self.priorEpsilonA + 0.5 * torch.sum(input=Phi, dim=0)

        # print('迭代出来的样子')
        # print(variationalEpsilonA.shape)

        variationalEpsilonBError = outputData.unsqueeze(0).repeat(self.variationalT, 1, 1) - torch.bmm(input=xTil, mat2=variationalWM)

        variationalEpsilonBTemp1 = torch.bmm(input=variationalEpsilonBError.permute(0, 2, 1), mat2=torch.diag_embed(input=Phi.permute(1, 0), offset=0, dim1=0).permute(1, 0, 2))
        variationalEpsilonBTemp1 = torch.bmm(input=variationalEpsilonBTemp1, mat2=variationalEpsilonBError).permute(1, 0, 2).squeeze(2)



        variationalEpsilonBTemp2 = torch.bmm(input=xTil.permute(0, 2, 1), mat2=torch.diag_embed(input=Phi.permute(1, 0), offset=0, dim1=0).permute(1, 0, 2))
        variationalEpsilonBTemp2 = torch.bmm(input=variationalEpsilonBTemp2, mat2=xTil)
        variationalEpsilonBTemp2 = torch.bmm(input=variationalEpsilonBTemp2, mat2=variationalWS)
        variationalEpsilonBTemp2 = torch.diagonal(variationalEpsilonBTemp2, dim1=-2, dim2=-1).sum(-1).unsqueeze(0)


        variationalEpsilonB = self.priorEpsilonB + 0.5 * (variationalEpsilonBTemp1 + variationalEpsilonBTemp2)



        return BetaGamma1, BetaGamma2, variationalGaussianMeanMu, variationalGaussianVarKappa, variationalWishartNu, variationalWishartPsi, variationalWM, variationalWS, variationalZetaA, variationalZetaB, variationalEpsilonA, variationalEpsilonB

    def vbEStep(self, BetaGamma1, BetaGamma2, variationalWishartPsi, variationalWishartNu, variationalGaussianMeanMu, variationalGaussianVarKappa, inputData):

        # 计算第1项 维度: (n, t)
        tempPhiTerm1 = (torch.digamma(input=(BetaGamma1)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).repeat(self.dataNumber, 1)

        # 计算第2项 维度: (n, t)
        tempPhiTerm2 = (torch.digamma(input=(BetaGamma2)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).T

        tempPhiTerm2 = tempPhiTerm2.repeat(1, self.variationalT) * torch.triu(input=torch.ones([self.variationalT, self.variationalT], device=self.device), diagonal=1)

        tempPhiTerm2 = torch.sum(input=tempPhiTerm2, dim=0, keepdim=True).repeat(self.dataNumber, 1)

        # 计算第3项 维度: (t, d, d)
        tempEqEta1 = ((variationalWishartNu.T.unsqueeze(2).repeat(1, self.inputDim, self.inputDim)) * torch.inverse(input=variationalWishartPsi.permute(2, 0, 1))).view(self.variationalT, self.inputDim * self.inputDim)

        # 计算第3项 维度: (t, d, 1)
        tempEqEta2 = variationalWishartNu.T.unsqueeze(2).repeat(1, self.inputDim, self.inputDim) * torch.inverse(input=variationalWishartPsi.permute(2, 0, 1))
        tempEqEta2 = torch.bmm(input=tempEqEta2, mat2=variationalGaussianMeanMu.T.unsqueeze(2))

        # 计算第3项 维度: (1, t)
        tempEqEta3 = -1.0 * torch.digamma(input=0.5 * variationalWishartNu) - self.inputDim * math.log(2.0) + torch.log(torch.det(variationalWishartPsi.permute(2, 0, 1)))

        #计算第4项

        tempEqEta42 = torch.bmm(input=variationalGaussianMeanMu.T.unsqueeze(1), mat2=torch.inverse(variationalWishartPsi.permute(2, 0, 1)))
        tempEqEta42 = torch.bmm(input=tempEqEta42, mat2=variationalGaussianMeanMu.T.unsqueeze(2)).squeeze(2)

        # 真正的第4项 维度: (1, t)
        tempEqEta4 = -0.5 * float(self.inputDim) / (variationalGaussianVarKappa) -0.5 * variationalWishartNu * tempEqEta42.T


        tempPhiTerm311 = (tempEqEta1.view(self.variationalT, self.inputDim * self.inputDim)).unsqueeze(0).repeat(self.dataNumber, 1, 1)
        tempPhiTerm312 = (-0.5 * torch.bmm(input=inputData.unsqueeze(2), mat2=inputData.unsqueeze(1))).view(self.dataNumber, self.inputDim * self.inputDim)
        # 第3项第1个 维度: (n, t)
        tempPhiTerm31 = torch.bmm(input=tempPhiTerm311, mat2=tempPhiTerm312.unsqueeze(2)).squeeze(2)

        # 第3项第2个 维度: (n, t)
        tempPhiTerm32 = tempEqEta2.squeeze(2) @ inputData.T
        # 第3项第3个 维度: (n, t)
        tempPhiTerm33 = -0.5 * tempEqEta3.repeat(self.dataNumber, 1).T

        # 第4项 维度: (n, t)
        tempPhiTerm4 = tempEqEta4.repeat(self.dataNumber, 1)



        finalPhi = tempPhiTerm1 + tempPhiTerm2 + tempPhiTerm31 + tempPhiTerm32.T + tempPhiTerm33.T + tempPhiTerm4


        finalPhiExp = torch.exp(input=finalPhi)

        finalPhiExp = finalPhiExp / torch.sum(input=finalPhiExp, dim=1, keepdim=True).repeat(1, finalPhiExp.shape[1])

        return finalPhiExp

        # 第四个自然参数的计算

    def truncation(self, threshold=0.1):
        # 进行截断操作
        piList = torch.sum(input=self.variationalPhi, dim=0, keepdim=True) / torch.sum(input=self.variationalPhi, dim=[0, 1])
        self.piList = piList
        piListTruncation = piList[piList >= threshold]
        self.clusterNum = piListTruncation.shape[0]

        # torch.sort(input, dim=-1, descending=False, stable=False, *, out=None)
        piList, piIdx = torch.sort(input=piList, dim=-1, descending=True)
        piIdx = piIdx.squeeze(0)

        # 进行重新排序
        self.variationalBetaGamma1 = self.variationalBetaGamma1[:, piIdx]
        self.variationalBetaGamma2 = self.variationalBetaGamma2[:, piIdx]

        # 高斯的变分超参数
        self.variationalGaussianMeanMu = self.variationalGaussianMeanMu[:, piIdx]
        self.variationalGaussianVarKappa = self.variationalGaussianVarKappa[:, piIdx]

        # 威沙特的变分超参数
        self.variationalWishartNu = self.variationalWishartNu[:, piIdx]
        self.variationalWishartPsi = self.variationalWishartPsi[:, :, piIdx]

        # 变分超参数Phi
        self.variationalPhi = self.variationalPhi[:, piIdx]

        # 变分超参数

        # 权重参数
        self.variationalWM = self.variationalWM[:, piIdx]
        self.variationalWS = self.variationalWS[:, :, piIdx]

        # 权重的方差
        self.variationalZetaA = self.variationalZetaA[:, piIdx]
        self.variationalZetaB = self.variationalZetaB[:, piIdx]

        # 残差的分布超参数
        self.variationalEpsilonA = self.variationalEpsilonA[:, piIdx]
        self.variationalEpsilonB = self.variationalEpsilonB[:, piIdx]


        # 存储变分超参数
        self.variationalBetaGamma1Old, self.variationalBetaGamma2Old = self.variationalBetaGamma1, self.variationalBetaGamma2
        self.variationalGaussianMeanMuOld, self.variationalGaussianVarKappaOld = self.variationalGaussianMeanMu, self.variationalGaussianVarKappa
        self.variationalWishartNuOld, self.variationalWishartPsiOld = self.variationalWishartNu, self.variationalWishartPsi
        self.variationalPhiOld = self.variationalPhi
        self.variationalWMOld, self.variationalWSOld = self.variationalWM, self.variationalWS
        self.variationalZetaAOld, self.variationalZetaBOld = self.variationalZetaA, self.variationalZetaB
        self.variationalEpsilonAOld, self.variationalEpsilonBOld = self.variationalEpsilonA, self.variationalEpsilonB



        # 截断超参数列表
        piIdxList = torch.tensor(np.arange(self.clusterNum), dtype=torch.long)

        # 进行截断操作

        # Beta的变分超参数
        self.variationalBetaGamma1 = self.variationalBetaGamma1[:, piIdxList]
        self.variationalBetaGamma2 = self.variationalBetaGamma2[:, piIdxList]

        # 高斯的变分超参数
        self.variationalGaussianMeanMu = self.variationalGaussianMeanMu[:, piIdxList]
        self.variationalGaussianVarKappa = self.variationalGaussianVarKappa[:, piIdxList]

        # 威沙特的变分超参数
        self.variationalWishartNu = self.variationalWishartNu[:, piIdxList]
        self.variationalWishartPsi = self.variationalWishartPsi[:, :, piIdxList]

        # 变分超参数Phi
        self.variationalPhi = self.variationalPhi[:, piIdxList]

        # 权重参数
        print('超参数截断')
        print(self.variationalWM.shape)

        self.variationalWM = self.variationalWM[:, piIdxList]
        print(self.variationalWM.shape)
        self.variationalWS = self.variationalWS[:, :, piIdxList]

        # 权重的方差
        self.variationalZetaA = self.variationalZetaA[:, piIdxList]
        self.variationalZetaB = self.variationalZetaB[:, piIdxList]

        # 残差的分布超参数
        self.variationalEpsilonA = self.variationalEpsilonA[:, piIdxList]
        self.variationalEpsilonB = self.variationalEpsilonB[:, piIdxList]
        return self

    def classPrediction(self):
        predictionClass = torch.argmax(input=self.variationalPhi, dim=1)

        predictionClass = pd.DataFrame(data=predictionClass.cpu().numpy())
        return predictionClass

    def vbEMAlgo(self, epoch=50):
        # BetaGamma1, BetaGamma2, variationalGaussianMeanMu, variationalGaussianVarKappa, variationalWishartNu, variationalWishartPsi

        t = time.time()
        for i in range(epoch):
            i = i + 1
            # BetaGamma1, BetaGamma2, variationalGaussianMeanMu, variationalGaussianVarKappa, variationalWishartNu, variationalWishartPsi
            self.variationalBetaGamma1, self.variationalBetaGamma2, \
            self.variationalGaussianMeanMu, self.variationalGaussianVarKappa, \
            self.variationalWishartNu, self.variationalWishartPsi, \
            self.variationalWM, self.variationalWS, \
            self.variationalZetaA, self.variationalZetaB, \
            self.variationalEpsilonA, self.variationalEpsilonB = self.vbMStep(BetaGamma1=self.priorBetaGamma1,
                                                                    Phi=self.variationalPhi, inputData=self.inputData, outputData=self.outputData)

            self.variationalPhi = self.vbEStep(BetaGamma1=self.variationalBetaGamma1,
                                               BetaGamma2=self.variationalBetaGamma2,
                                               variationalWishartPsi=self.variationalWishartPsi,
                                               variationalWishartNu=self.variationalWishartNu,
                                               variationalGaussianMeanMu=self.variationalGaussianMeanMu,
                                               variationalGaussianVarKappa=self.variationalGaussianVarKappa, inputData=self.inputData)

        self.runningTime = time.time() - t

        print('variationalEM迭代出来的样子')
        print(self.variationalWM.shape)
        self.variationalWM = self.variationalWM.squeeze(2).permute(1, 0)

        self.variationalWS = self.variationalWS.permute(1, 2, 0)
        print(self.variationalWS.shape)
        print('variational Bayesian running time:{} s'.format(str(self.runningTime)))

        return self

    def prediction(self, dataX, BetaGamma1, BetaGamma2, variationalWishartPsi, variationalWishartNu, variationalGaussianMeanMu, variationalGaussianVarKappa):

        dataX = torch.tensor(data=dataX, dtype=torch.float32, device=self.device)
        dataXNumber, dimX = dataX.shape[0], dataX.shape[1]
        variationalGaussianMeanMu = variationalGaussianMeanMu[0: dimX, :]
        variationalWishartPsi = variationalWishartPsi[0: dimX, 0: dimX, :]
        # 计算第1项 维度: (n, t)
        tempPhiTerm1 = (torch.digamma(input=(BetaGamma1)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).repeat(
            dataXNumber, 1)

        # 计算第2项 维度: (n, t)
        tempPhiTerm2 = (torch.digamma(input=(BetaGamma2)) - torch.digamma(input=(BetaGamma1 + BetaGamma2))).T

        tempPhiTerm2 = tempPhiTerm2.repeat(1, self.clusterNum) * torch.triu(
            input=torch.ones([self.clusterNum, self.clusterNum], device=self.device), diagonal=1)

        tempPhiTerm2 = torch.sum(input=tempPhiTerm2, dim=0, keepdim=True).repeat(dataXNumber, 1)

        # 计算第3项 维度: (t, d, d)
        tempEqEta1 = ((variationalWishartNu.T.unsqueeze(2).repeat(1, dimX, dimX)) * torch.inverse(
            input=variationalWishartPsi.permute(2, 0, 1))).view(self.clusterNum, dimX * dimX)

        # 计算第3项 维度: (t, d, 1)
        tempEqEta2 = variationalWishartNu.T.unsqueeze(2).repeat(1, dimX, dimX) * torch.inverse(
            input=variationalWishartPsi.permute(2, 0, 1))
        tempEqEta2 = torch.bmm(input=tempEqEta2, mat2=variationalGaussianMeanMu.T.unsqueeze(2))

        # 计算第3项 维度: (1, t)
        tempEqEta3 = -1.0 * torch.digamma(input=0.5 * variationalWishartNu) - dimX * math.log(2.0) + torch.log(
            torch.det(variationalWishartPsi.permute(2, 0, 1)))

        # 计算第4项

        tempEqEta42 = torch.bmm(input=variationalGaussianMeanMu.T.unsqueeze(1),
                                mat2=torch.inverse(variationalWishartPsi.permute(2, 0, 1)))
        tempEqEta42 = torch.bmm(input=tempEqEta42, mat2=variationalGaussianMeanMu.T.unsqueeze(2)).squeeze(2)

        # 真正的第4项 维度: (1, t)
        tempEqEta4 = -0.5 * float(dimX) / (
            variationalGaussianVarKappa) - 0.5 * variationalWishartNu * tempEqEta42.T

        tempPhiTerm311 = (tempEqEta1.view(self.clusterNum, dimX * dimX)).unsqueeze(0).repeat(dataXNumber, 1, 1)
        tempPhiTerm312 = (-0.5 * torch.bmm(input=dataX.unsqueeze(2), mat2=dataX.unsqueeze(1))).view(dataXNumber,
                                                                                                    dimX * dimX)
        # 第3项第1个 维度: (n, t)
        tempPhiTerm31 = torch.bmm(input=tempPhiTerm311, mat2=tempPhiTerm312.unsqueeze(2)).squeeze(2)

        # 第3项第2个 维度: (n, t)
        tempPhiTerm32 = tempEqEta2.squeeze(2) @ dataX.T
        # 第3项第3个 维度: (n, t)
        tempPhiTerm33 = -0.5 * tempEqEta3.repeat(dataXNumber, 1).T

        # 第4项 维度: (n, t)
        tempPhiTerm4 = tempEqEta4.repeat(dataXNumber, 1)

        finalPhi = tempPhiTerm1 + tempPhiTerm2 + tempPhiTerm31 + tempPhiTerm32.T + tempPhiTerm33.T + tempPhiTerm4

        finalPhiExp = torch.exp(input=finalPhi)

        finalPhiExp = finalPhiExp / torch.sum(input=finalPhiExp, dim=1, keepdim=True).repeat(1, finalPhiExp.shape[1])



        dataXMap = self.mapFunction(inputData=dataX)
        xForPred = torch.cat((dataXMap, torch.ones((dataX.shape[0], 1), dtype=torch.float32, device=self.device)), dim=1).unsqueeze(0).repeat(self.clusterNum, 1, 1)
        # xForPred = torch.cat((dataX, torch.ones((dataX.shape[0], 1), dtype=torch.float32, device=self.device))

        dataPred = torch.bmm(input=xForPred, mat2=self.variationalWM.unsqueeze(1).permute(2, 0, 1))
        # print('打印形态')
        # print(finalPhiExp.shape)
        # print(dataPred.shape)
        #
        # print(finalPhiExp.shape)

        # sys.exit(0)

        print('检查是否有脏数据')
        print(torch.sum(input=(dataXMap != dataXMap)))
        print(torch.sum(input=(finalPhi != finalPhi)))
        print(finalPhi)
        print(torch.sum(input=(finalPhiExp != finalPhiExp)))

        yPred = torch.sum(input=dataPred * finalPhiExp.permute(1, 0).unsqueeze(2), dim=0, keepdim=False).cpu()


        return yPred



#
# data = pd.read_csv('iris.csv', sep=',', header=None, index_col=None)
# data = data.iloc[:, 1:3]
# print(data)

# 开始干活儿
data = pd.read_csv('CO2Absorber_Train.csv', sep=',', header=0, index_col=0)

# data = pd.read_csv('CO2Absorber_Test.csv', sep=',', header=0, index_col=0)

scalerX = StandardScaler()
scalerY = StandardScaler()

print('打印数据')
print(data)

# print(data.iloc[:, 0:11])

dataX = np.array(data.iloc[:, 0:11])
dataX = scalerX.fit_transform(dataX)
dataY = np.array(data.iloc[:, -1]).reshape((-1, 1))
dataY = scalerY.fit_transform(dataY)
dataFinal = np.hstack((dataX, dataY))


# print(data)
# GPU不支持批次矩阵求逆操作
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

DPMMTest1 = DPMM(variationalT=10, alphaDP=1.0 / 1000.0, data=dataFinal, device=DEVICE, inputDim=11).to(DEVICE).vbEMAlgo(epoch=150)



DPMMTest1.truncation(threshold=0.00)

print('Before Truncation:')
print(DPMMTest1.variationalT)
print('After Truncation:')
print(DPMMTest1.clusterNum)
print('Phi value list:')
print(DPMMTest1.piList)





#
dataTest = pd.read_csv('CO2Absorber_Test.csv', sep=',', header=0, index_col=0)
# dataTest = pd.read_csv('CO2Absorber_Train.csv', sep=',', header=0, index_col=0)
testX = np.array(dataTest.iloc[:, 0:11])
testX = scalerX.transform(testX)

yPred = DPMMTest1.prediction(dataX=testX,
                             BetaGamma1=DPMMTest1.variationalBetaGamma1,
                             BetaGamma2=DPMMTest1.variationalBetaGamma2,
                             variationalWishartPsi=DPMMTest1.variationalWishartPsi,
                             variationalWishartNu=DPMMTest1.variationalWishartNu,
                             variationalGaussianMeanMu=DPMMTest1.variationalGaussianMeanMu,
                             variationalGaussianVarKappa=DPMMTest1.variationalGaussianVarKappa)

yPred = scalerY.inverse_transform(yPred)
# yPredDF = pd.DataFrame(yPred)
# yPredDF.to_csv('prediction.csv', sep=',')
# print(yPred)
#
#
#
dataY = np.array(dataTest.iloc[:, -1]).reshape((-1, 1))

print(DPMMTest1.variationalWM)
r2 = r2_score(dataY, yPred)
rmse = math.sqrt(mean_squared_error(dataY, yPred))
print('Valuation Indices, r2: {}, rmse: {}'.format(r2, rmse))

print('打印形态')
print(yPred.shape[0])
# 测试集画图
plt.figure()
plt.plot(range(yPred.shape[0]), dataY[:, 0], color='r', label='y_true')
plt.plot(range(yPred.shape[0]), yPred[:, 0], color='b', label='y_testpre')

plt.legend()
plt.show()