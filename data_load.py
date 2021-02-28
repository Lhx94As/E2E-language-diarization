import os
import glob
import scipy
import shutil
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
from soundfile import SoundFile


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    seq, label = zip(*batch)
    seq_length = [len(x) for x in label]
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    # label_stack = []
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0)
    # return data, torch.tensor(label_stack), seq_length
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
        # feature = np.load(feature_path, allow_pickle=True)
        label = [int(x) for x in self.label_list[index]]
        # label = torch.zeros((len(labels_), 3))
        # for ind_ in range(len(labels_)):
        #     ind = labels_[ind_]
        #     label[ind_, ind] = 1
        return feature, torch.LongTensor(label)
        # return feature, label

    def __len__(self):
        return len(self.label_list)

def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    # assert len(seq_lens) != batch_size
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[:length,:length] = 0
    return atten_mask.bool()


if __name__ == "__main__":
    from model import BLSTM_E2E_LID
    from torch.utils.data import DataLoader

    # data_dir = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
    #            'PartB_Tamil/PartB_Tamil/Train/toy.txt'
    #
    #
    # dataset = RawFeatures(data_dir)
    # data = DataLoader(dataset=dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    # for i, j, k in data:
    #     # print("original:",i)
    #     # print("original label:",j)
    #     # zz = rnn_utils.pack_padded_sequence(i,k,batch_first=True)
    #     # print("pack:",zz.data)
    #     ll = rnn_utils.pack_padded_sequence(j,k,batch_first=True)
    #     # ll_ = torch.transpose(ll,)
    #     print("pack label:",ll.data)
    #     ll_ = rnn_utils.pad_packed_sequence(ll, batch_first=True)
    #     print(ll_)



    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # net = BLSTM_E2E_LID(input_dim=3).to(device)
    # net.train()
    # emb, out = net(input_data)
    # kk = torch.split(out, b)
    # print(kk[0])
    # print(kk[1])

    # print(out)
    # # print(out_pad)
    #
    # # print(a)
    # # print(rnn_utils.pack_padded_sequence(a,b,batch_first=True))
    lstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=1, batch_first=True)
    linear = nn.Linear(3,2)
    input_1 = (torch.FloatTensor([[1,1,1],[70,8,9]]),torch.FloatTensor([[70,8,9]]))
    input_2 = torch.FloatTensor([[1,1,1],[70,8,9]])
    label1 = (torch.LongTensor([0,1]),torch.LongTensor([1]))
    label1 = rnn_utils.pad_sequence(label1)
    label1_ = rnn_utils.pack_padded_sequence(label1,[2,1]).data
    label2 = torch.LongTensor([0,1])

    data = rnn_utils.pad_sequence(input_1)
    # output1,_ = lstm(data)
    output1_,_ = lstm(rnn_utils.pack_padded_sequence(data,[2,1],batch_first=True))
    output1 = rnn_utils.pad_packed_sequence(output1_, batch_first=True)
    # print(output1[0])
    output1 = linear(output1[0])
    # output1_ = linear(output1_.data)
    # # print(rnn_utils.pack_padded_sequence(data,[2,1]))
    # output2,_ = lstm(input_2.view(1,2,3))
    # output2 = linear(output2)
    # loss = nn.CrossEntropyLoss()
    # # print(loss(output1,label1))
    # # print(loss(output1_,label1_))
    # print(label1_)
    print(output1)
    # print(output1_)
    # # print(output2)


