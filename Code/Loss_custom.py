import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg
import pdb
from scipy.stats import pearsonr


def signal_corelation(signal1, signal2):
    if np.all(signal1 == 0) or np.all(signal2 == 0):
        pcc = 0
    else:
        pcc, p_value = pearsonr(signal1, signal2)
    return pcc


def P_MSE(gen, real):
    """Persent mean square error
    """
    rate = 10.0

    row = gen.shape[1]
    col = gen.shape[2]
    batch = gen.shape[0]
    thres = torch.full([row, col], 0.01).to(gen.device).float()
    nonzero_real = torch.where(0.0 != real, real, thres).to(gen.device).float()
    mul = (gen - real).to(gen.device).float()
    per = torch.div(mul, nonzero_real).to(gen.device).float()
    loss = torch.sum(torch.abs(per))
    loss = torch.div(loss, float(batch)) * rate

    return loss


def main_PMSE():
    a = torch.full([2, 3, 3], -2).float()
    b = torch.randint(-2, 3, (2, 3, 3)).float()
    loss = P_MSE(a, b)
    # print loss


def Pearson_loss_regions(gen, real):
    batch = gen.shape[0]
    loss = 0.0
    eps = 1e-6
    for i in range(batch):
        region_num = gen[i].shape[0]
        for region in range(region_num):
            gen_vec = gen[i][region].to(gen.device)
            real_vec = real[i][region].to(gen.device)
            gen_mean = gen_vec - torch.mean(gen_vec) + eps
            real_mean = real_vec - torch.mean(real_vec) + eps
            r_num = torch.sum(gen_mean * real_mean)
            r_den = torch.sqrt(torch.sum(torch.pow(gen_mean, 2)) * torch.sum(torch.pow(real_mean, 2)))
            pear = r_num / r_den
            # pear_official = signal_corelation(gen_vec.numpy(), real_vec.numpy())
            # print pear, pear_official
            loss = loss + torch.pow(pear - 1.0, 2)
            # print loss
    loss = torch.div(loss, float(batch))
    return loss


def Pearson_loss_whole(gen, real):
    batch = gen.shape[0]
    loss = 0.0
    eps = 1e-6
    rate = 1.0
    for i in range(batch):
        gen_vec = gen[i].view(-1).to(gen.device)
        real_vec = real[i].view(-1).to(gen.device)
        gen_mean = gen_vec - torch.mean(gen_vec) + eps
        real_mean = real_vec - torch.mean(real_vec) + eps
        r_num = torch.sum(gen_mean * real_mean)
        r_den = torch.sqrt(torch.sum(torch.pow(gen_mean, 2)) * torch.sum(torch.pow(real_mean, 2)))
        pear = r_num / r_den
        loss = loss + torch.pow(pear - 1.0, 2)
    loss = torch.div(loss, float(batch)) * rate
    return loss


def main_Pearson_loss():
    a = torch.randn(6,120,120).float()
    b = torch.randn(6,120,120).float()
    loss = Pearson_loss_whole(a, b)
    print(loss)

    a = torch.randn(6, 120, 120).float()
    b = torch.randn(6, 120, 120).float()
    loss = Pearson_loss_regions(a, b)
    print(loss)


if __name__ == '__main__':
    main_Pearson_loss()








