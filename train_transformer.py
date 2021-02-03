import os
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from Loss import *
from data_load import *
from model_evaluation import *


def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_,output), dim=0)
    return output_

n_lang = 3
# model = BLSTM_E2E_LID(n_lang=n_lang,
#                       dropout=0.25,
#                       input_dim=437,
#                       hidden_size=256,
#                       num_emb_layer=2,
#                       num_lstm_layer=3,
#                       emb_dim=256)

model = Transformer_E2E_LID(n_lang=n_lang,
                            dropout=0.1,
                            input_dim=437,
                            feat_dim=256,
                            n_heads=4,
                            d_k=256,
                            d_v=256,
                            d_ff=2048)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to(device)

# train_txt = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
#             'PartB_Telugu/PartB_Telugu/Train/utt2lan.txt'
# train_txt = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
#             'PartB_Tamil/PartB_Tamil/Train/utt2lan.txt'
train_txt = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
            'PartB_Gujarati/PartB_Gujarati//Train/utt2lan.txt'
train_set = RawFeatures(train_txt)

# valid_set = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
#             'PartB_Telugu/PartB_Telugu/Dev/utt2lan.txt'
# valid_set = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
#             'PartB_Tamil/PartB_Tamil/Dev/utt2lan.txt'
valid_set = '/home/hexin/Desktop/hexin/datasets/First_workshop_codeswitching/' \
            'PartB_Gujarati/PartB_Gujarati/Dev/utt2lan.txt'
valid_set = RawFeatures(train_txt)

batch_size = 128
num_epoch = 100
train_data = DataLoader(dataset=train_set,
                        batch_size=batch_size,
                        pin_memory=True,
                        num_workers=16,
                        shuffle=True,
                        collate_fn=collate_fn_atten)

valid_data = DataLoader(dataset=valid_set,
                        batch_size=1,
                        pin_memory=True,
                        # num_workers=16,
                        shuffle=False,
                        collate_fn=collate_fn_atten)

loss_func_CRE = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
T_max = num_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

# Train the model
total_step = len(train_data)
best_acc = 0

for epoch in tqdm(range(num_epoch)):
    loss_item = 0
    model.train()
    for step, (utt, labels, seq_len) in enumerate(train_data):
        utt_ = utt.to(device=device, dtype=torch.float)
        atten_mask = get_atten_mask(seq_len, utt_.size(0))
        atten_mask = atten_mask.to(device=device)
        labels = labels.to(device=device,dtype=torch.long)
        # Forward pass
        outputs = model(utt_, seq_len,atten_mask)
        outputs = get_output(outputs, seq_len)
        loss = loss_func_CRE(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epoch, step + 1, total_step, loss.item()))
torch.save(model.state_dict(), '/home/hexin/Desktop/models/' + '{}.ckpt'.format("Transformer"))
model.eval()
correct = 0
total = 0
predicts = []
FAR_list = torch.zeros(n_lang)
FRR_list = torch.zeros(n_lang)
eer = 0
for step, (utt, labels, seq_len) in enumerate(valid_data):
    utt_ = utt.to(device=device, dtype=torch.float)
    labels = labels.to(device=device, dtype=torch.long)
    # Forward pass\
    outputs = model(x=utt_,seq_len=seq_len, atten_mask=None)
    outputs = get_output(outputs, seq_len)
    # print(outputs)
    predicted = torch.argmax(outputs,-1)
    total += labels.size(-1)
    correct += (predicted == labels).sum().item()
    FAR, FRR = compute_far_frr(n_lang, predicted, labels)
    FAR_list += FAR
    FRR_list += FRR
acc = correct/total
print('Current Acc.: {:.4f} %'.format(100 * acc))
for i in range(n_lang):
    eer_ = (FAR_list[i]/total + FRR_list[i]/total)/2
    eer += eer_
    print("EER for label {}: {:.4f}%".format(i, eer_*100))
print('EER: {:.4f} %'.format(100*eer/n_lang))
# print('Val Loss: {:.4f}'.format(loss_test.item()))
