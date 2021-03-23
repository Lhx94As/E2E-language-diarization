import os
import glob
import scipy
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0)
    return data, label, seq_length


def collate_fn_atten(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = 0
    for i in range(len(label)):
        if i == 0:
            labels = label[i]
        else:
            labels = torch.cat((labels, label[i]),-1)
    return data, labels, seq_length


def collate_fn_cnn_atten(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    label_cnn = rnn_utils.pad_sequence(label, batch_first=True, padding_value=255)
    labels = 0
    label_cnn_ = 0
    for i in range(len(label)):
        if i == 0:
            labels = label[i]
            label_cnn_ = label_cnn[0]
        else:
            labels = torch.cat((labels, label[i]),-1)
            label_cnn_ = torch.cat((label_cnn_, label_cnn[i]),-1)
    return data, labels, label_cnn_, seq_length


class RawFeatures(data.Dataset):
    def __init__(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.feature_list = [i.split()[0] for i in lines]
            self.label_list = [i.split()[-1] for i in lines]

    def __getitem__(self, index):
        feature_path = self.feature_list[index]
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True))
        label = [int(x) for x in self.label_list[index]]
        return feature, torch.LongTensor(label)
        # return feature, label

    def __len__(self):
        return len(self.label_list)

def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[:length,:length] = 0
    return atten_mask.bool()


