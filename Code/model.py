import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pdb


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__()

        self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batchnorm_layer(x)
        x = x.permute(0, 2, 1)
        return x


class BatchNormLinear(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormLinear, self).__init__()

        self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = self.batchnorm_layer(x)
        return x


class BatchNormAdj(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormAdj, self).__init__()
        self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        batch_size = x.size(0)
        num_region = x.size(1)
        x = x.contiguous().view(batch_size, -1)
        x = self.batchnorm_layer(x)
        x = x.contiguous().view(batch_size, num_region, -1)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, elementwise_affine=True):
        super(LayerNorm, self).__init__()

        self.LayerNorm = nn.LayerNorm(num_features, eps, elementwise_affine)#num_features = (input.size()[1:])

    def forward(self, x):
        x = self.LayerNorm(x)
        return x


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.einsum('bij,bjd->bid', [adj, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNGenerator(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, out3_feature, dropout):
        super(GCNGenerator, self).__init__()

        self.gc1 = GraphConvolution(in_feature, in_feature)
        self.LayerNorm1 = LayerNorm([in_feature, in_feature])

        self.gc2_01 = GraphConvolution(in_feature, int(in_feature*2))
        self.LayerNorm2_01 = LayerNorm([in_feature, int(in_feature*2)])
        self.gc2_12 = GraphConvolution(int(in_feature*2), in_feature)
        self.LayerNorm2_12 = LayerNorm([in_feature, in_feature])

        self.gc3_01 = GraphConvolution(in_feature, int(in_feature/2))
        self.LayerNorm3_01 = LayerNorm([in_feature, int(in_feature/2)])
        self.gc3_13 = GraphConvolution(int(in_feature/2), in_feature)
        self.LayerNorm3_13 = LayerNorm([in_feature, in_feature])

        # the theta: can compare different initializations
        self.weight = Parameter(torch.FloatTensor([0.0, 0.0, 0.0, ]))


    def forward(self, topo, funcs, batchSize, isTest=False):

        topo = funcs # to compare with different updating methods: topo != funcs 

        x1 = self.gc1(funcs, topo)
        x1 = self.LayerNorm1(x1)
        x1 = F.leaky_relu(x1, 0.05, inplace=True)

        x2 = self.gc2_01(funcs, topo)
        x2 = self.LayerNorm2_01(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=True)
        x2 = self.gc2_12(x2, topo)
        x2 = self.LayerNorm2_12(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=True)

        x3 = self.gc3_01(funcs, topo)
        x3 = self.LayerNorm3_01(x3)
        x3 = F.leaky_relu(x3, 0.05, inplace=True)
        x3 = self.gc3_13(x3, topo)
        x3 = self.LayerNorm3_13(x3)
        x3 = F.leaky_relu(x3, 0.05, inplace=True)

        x = self.weight[0]*x1 + self.weight[1]*x2 + self.weight[2]*x3
        outputs = x + torch.transpose(x, 1, 2)
        if isTest is True:
            return outputs.squeeze().unsqueeze(0)
        else:
            return outputs.squeeze()


class CNNGenerator1(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, out3_feature, dropout):
        super(CNNGenerator1, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.weight1 = Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25, ]))
        self.weight2 = Parameter(torch.FloatTensor([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, ]))
        self.LayerNorm = LayerNorm([in_feature, in_feature])

    def forward(self, topo, func_matrix, batchSize, isTest=False):
        # topo not used in CNN based generator
        func_matrix = func_matrix.unsqueeze(1)
        x0 = torch.cat((func_matrix, func_matrix), 1)

        x1 = self.conv1(x0)
        x1 = self.LayerNorm(x1)
        x1 = F.leaky_relu(x1, 0.5, inplace=True)
        x1 = self.weight1[0] * x1[:, 0] + self.weight1[1] * x1[:, 1] + self.weight1[2] * x1[:, 2] + self.weight1[3] * x1[:, 3]

        x2 = self.conv2(x0)
        x2 = self.LayerNorm(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=True)
        x2 = self.weight2[0] * x2[:, 0] + self.weight2[1] * x2[:, 1] + self.weight2[2] * x2[:, 2] + self.weight2[3] * x2[:, 3] + self.weight2[4] * x2[:, 4] + self.weight2[5] * x2[:, 5] + self.weight2[6] * x2[:, 6] + self.weight2[7] * x2[:, 7]

        x = x1 + x2
        outputs = F.leaky_relu(x, 0.5, inplace=True)
        outputs = outputs + torch.transpose(outputs, 1, 2)
        if isTest is True:
            return outputs.squeeze().unsqueeze(0)
        else:
            return outputs.squeeze()


class CNNGenerator2(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, out3_feature, dropout):
        super(CNNGenerator2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.dropout = nn.Dropout(dropout)
        self.Linear1 = nn.Linear(3*3*256, in_feature)
        self.Linear2 = nn.Linear(3*3*256, in_feature)


    def forward(self, topo, func_matrix, batchSize, isTest=False):

        x = func_matrix.unsqueeze(1)
        x = self.features(x)

        x = torch.flatten(x, 1)

        x = torch.bmm(self.Linear1(x).unsqueeze(2), self.Linear2(x).unsqueeze(1))

        outputs = x + torch.transpose(x, 1, 2)
        if isTest is True:
            return outputs.squeeze().unsqueeze(0)
        else:
            return outputs.squeeze()


class Discriminator(nn.Module):
    def __init__(self, in_feature, out1_feature, out2_feature, out3_feature, dropout):
        super(Discriminator, self).__init__()

        self.gc1 = GraphConvolution(in_feature, out1_feature)
        self.batchnorm1 = BatchNorm(out1_feature)
        self.gc2 = GraphConvolution(out1_feature, out2_feature)
        self.batchnorm2 = BatchNorm(out2_feature)
        self.gc3 = GraphConvolution(out2_feature, out3_feature)
        self.batchnorm3 = BatchNorm(out3_feature)
        self.batchnorm4 = BatchNormLinear(1024)
        self.dropout = dropout
        self.Linear1 = nn.Linear(out3_feature * in_feature, 1024) # 148 for atlas1 and 68 for atlas2
        self.dropout = dropout
        self.Linear2 = nn.Linear(1024, 1)


    def batch_eye(self, size):
        batch_size = size[0]
        n = size[1]
        I = torch.eye(n).unsqueeze(0)
        I = I.repeat(batch_size, 1, 1)
        return I

    def forward(self, adj_matrix, batchSize, isTest=False):
        x = self.batch_eye(adj_matrix.shape).to(adj_matrix.device).float()

        x = self.gc1(x, adj_matrix)
        x = nn.LeakyReLU(0.2, True)(x)
        x = self.batchnorm1(x)

        x = self.gc2(x, adj_matrix)
        x = nn.LeakyReLU(0.2, True)(x)
        x = self.batchnorm2(x)

        x = self.gc3(x, adj_matrix)
        x = nn.LeakyReLU(0.2, True)(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.contiguous().view(batchSize, -1)
        x = self.Linear1(x)
        x = nn.LeakyReLU(0.2, True)(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Linear2(x)
        x = torch.sigmoid(x)
        x = x.contiguous().view(batchSize, -1)
        outputs = x
        return outputs.squeeze()