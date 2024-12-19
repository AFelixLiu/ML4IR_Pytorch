"""
@author AFelixLiu
@date 2024 12月 18
"""

import torch
from torch import nn


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    @staticmethod
    def forward(pred, label):
        # Calculate diff
        diff = pred - label

        # Calculate the absolute sum of the cumulative diff
        cumsum_diff = torch.cumsum(diff, dim=1)

        # 2选1 (推荐不取均值)
        # loss = torch.mean(torch.sum(torch.abs(cumsum_diff), dim=1))  # 结果为单个样本的loss
        loss = torch.sum(torch.abs(cumsum_diff))  # 结果为一个batch样本的sum_loss

        return loss


class Predictor(nn.Module):
    def __init__(self, input_dim, layers_dim, output_dim):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, layers_dim[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(0, len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(layers_dim[-1], output_dim))

        # Loss function (1/2)
        self.loss_1 = EMDLoss()

        # Loss function (2/2)
        self.loss_2 = nn.MSELoss(reduction='sum')

    def forward(self, X, Y):
        for layer in self.layers:
            X = layer(X)
        pred = torch.abs(X)

        normed_pred, normed_Y = self.normalize(pred, Y)
        emd_loss = self.loss_1(normed_pred, normed_Y)
        mse_loss = self.loss_2(normed_pred, normed_Y)

        return pred, emd_loss, mse_loss

    @staticmethod
    def normalize(A, B):
        """
        对输入的张量A和B进行标准化处理，确保数据类型正确以及维度合适
        """
        if isinstance(A, torch.Tensor):
            if A.dtype != torch.float:
                A = A.type(torch.float)
        else:
            A = torch.tensor(A, dtype=torch.float)

        if isinstance(B, torch.Tensor):
            if B.dtype != torch.float:
                B = B.type(torch.float)
        else:
            B = torch.tensor(B, dtype=torch.float)

        # 增加维度以适配后续按维度求和等操作（如果输入是一维tensor的情况）
        if A.dim() == 1:
            A = A.unsqueeze(0)
        if B.dim() == 1:
            B = B.unsqueeze(0)

        # 标准化
        normed_A = A / A.sum(dim=1, keepdim=True)
        normed_B = B / B.sum(dim=1, keepdim=True)

        return normed_A, normed_B
