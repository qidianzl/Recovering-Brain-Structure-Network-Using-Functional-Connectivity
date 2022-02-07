import torch.utils.data as data
import os
import random
import os.path
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
import matplotlib.pylab as plt
import seaborn as sns

import pdb


def subjects_error_list(subjects, data_path, empty_list):

    def zero_row_col(matrix, subject, subjects_error):
        row_zero = np.zeros(shape=(matrix.shape[1],))
        col_zero = np.zeros(shape=(matrix.shape[0],))
        row_index = []
        col_index = []

        for row in range(matrix.shape[0]):
            if (matrix[row] == row_zero).all():
                row_index.append(row)

        for col in range(matrix.shape[1]):
            if (matrix[:, col] == col_zero).all():
                col_index.append(col)

        if empty_list != sorted(row_index) or empty_list != sorted(col_index):
            if subject not in subjects_error:
                subjects_error.append(subject)

    subjects_error = []

    for subject in subjects:
        adj_matrix = np.loadtxt(data_path + '/' + subject + '/' + 'common_fiber_matrix.txt')
        func_matrix = np.loadtxt(data_path + '/' + subject + '/' + 'pcc_fmri_feature_matrix_0.txt')
        zero_row_col(adj_matrix, subject, subjects_error)
        zero_row_col(func_matrix, subject, subjects_error)

    return subjects_error


def adj_matrix_normlize(adj):
    adj_log = adj + 1
    adj_log = np.log2(adj_log)
    # adj_log_norm = (adj_log - adj_log.min())/(adj_log.max()-adj_log.min())
    # I = np.identity(adj.shape[0])
    # adj_log_norm_diag = I-(I-1)*adj_log_norm
    return adj_log


def load_data(data_path, empty_list):
    # pdb.set_trace()
    subjects = [sub for sub in os.listdir(data_path) if not sub.startswith('.')]
    subjects_error = subjects_error_list(subjects, data_path, empty_list)
    data = []
    SubjID_list = [subject for subject in subjects if subject not in subjects_error]

    # pdb.set_trace()
    for subject in SubjID_list:
        adj_matrix_path = data_path + '/' + subject + '/' + 'common_fiber_matrix.txt'
        func_matrix_path = data_path + '/' + subject + '/' + 'pcc_fmri_feature_matrix_0.txt'
        data.append((subject, adj_matrix_path, func_matrix_path))
    random.shuffle(data)
    return data


def normlize_data(data, empty_list):
    eps = 1e-9
    all_funcs = []
    all_adjs = []
    for index in range(len(data)):
        subject, adj_matrix_path, func_matrix_path = data[index]

        adj_matrix = np.loadtxt(adj_matrix_path)
        adj_matrix = np.delete(adj_matrix, empty_list, axis=0)
        adj_matrix = np.delete(adj_matrix, empty_list, axis=1)
        adj_matrix = adj_matrix_normlize(adj_matrix)

        func_matrix = np.loadtxt(func_matrix_path)
        func_matrix = np.delete(func_matrix, empty_list, axis=0)
        func_matrix = np.delete(func_matrix, empty_list, axis=1)

        all_adjs.append(adj_matrix)
        all_funcs.append(func_matrix)

    all_adjs = np.stack(all_adjs)
    all_funcs = np.stack(all_funcs)

    adj_mean = all_adjs.mean((0, 1, 2), keepdims=True).squeeze(0)
    adj_std = all_adjs.std((0, 1, 2), keepdims=True).squeeze(0)

    func_mean = all_funcs.mean((0, 1, 2), keepdims=True).squeeze(0)
    func_std = all_funcs.std((0, 1, 2), keepdims=True).squeeze(0)

    return (torch.from_numpy(adj_mean) + eps, torch.from_numpy(adj_std) + eps, torch.from_numpy(func_mean) + eps,
            torch.from_numpy(func_std) + eps)



class MICCAI(data.Dataset):
    def __init__(self, data_path, all_data, data_mean, empty_list, train=True, test=False):
        self.data_path = data_path
        self.train = train  # training set or val set
        self.test = test
        self.adj_mean, self.adj_std, self.feat_mean, self.feat_std = data_mean
        self.empty_list = empty_list

        # pdb.set_trace()
        if self.train:
            self.data = all_data[:500]
        elif not test:
            self.data = all_data[500:600]
        else:
            self.data = all_data[600:]

        random.shuffle(self.data)

    def __getitem__(self, index):
        subject, adj_matrix_path, func_matrix_path = self.data[index]

        adj_matrix = np.loadtxt(adj_matrix_path)
        adj_matrix = np.delete(adj_matrix, self.empty_list, axis=0)
        adj_matrix = np.delete(adj_matrix, self.empty_list, axis=1)
        adj_matrix = adj_matrix_normlize(adj_matrix)

        func_matrix = np.loadtxt(func_matrix_path)
        func_matrix = np.delete(func_matrix, self.empty_list, axis=0)
        func_matrix = np.delete(func_matrix, self.empty_list, axis=1)

        adj_matrix = torch.from_numpy(adj_matrix)
        adj_matrix = (adj_matrix - self.adj_mean) / self.adj_std

        func_matrix = torch.from_numpy(func_matrix)
        func_matrix = (func_matrix - self.feat_mean) / self.feat_std

        return subject, adj_matrix, func_matrix

    def debug_getitem__(self, index=0):
        # pdb.set_trace()
        subject, adj_matrix_path, func_matrix_path = self.data[index]

        adj_matrix = np.loadtxt(adj_matrix_path)
        adj_matrix = np.delete(adj_matrix, self.empty_list, axis=0)
        adj_matrix = np.delete(adj_matrix, self.empty_list, axis=1)
        # adj_matrix = adj_matrix_normlize(adj_matrix)

        func_matrix = np.loadtxt(func_matrix_path)
        func_matrix = np.delete(func_matrix, self.empty_list, axis=0)
        func_matrix = np.delete(func_matrix, self.empty_list, axis=1)

        adj_matrix = torch.from_numpy(adj_matrix)
        func_matrix = torch.from_numpy(func_matrix)

        pdb.set_trace()

        return subject, adj_matrix, func_matrix

    def __len__(self):
        return len(self.data)


def get_loader(data_path, all_data, data_mean, empty_list, training, test, batch_size=16, num_workers=4):
    dataset = MICCAI(data_path, all_data, data_mean, empty_list, training, test)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers)

    return data_loader


if __name__ == '__main__':
    data_path = './data/HCP_1064_matrix_atlas2'
    all_data = load_data(data_path)
    data_mean = normlize_data(all_data)
    # get_common_adj(all_data, data_mean)
    # dataset = MICCAI(data_path, all_data)
    # for i in range(len(dataset)):
    #     x, y, z = dataset.debug_getitem__(i)




