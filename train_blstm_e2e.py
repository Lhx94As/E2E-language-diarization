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

#python train_blstm_e2e.py --savedir "/home/hexin/Desktop/models" --train "/home/hexin/Desktop/data/train.txt" --test "/home/hexin/Desktop/data/test.txt"
#                          --seed 0 --device 0 --batch 8 --epochs 60 --dim 23 --lang 3 --model my_sa_e2e --lr 0.00001 --lambda 0.5

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--model', type=str, help='model name', default='my_BLSTM')
    parser.add_argument('--savedir', type=str, help='dir in which the trained model is saved')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    parser.add_argument('--test', type=str, help='testing data, in .txt')
    parser.add_argument('--seed', type=int, help='Device name', default=0)
    parser.add_argument('--batch', type=int, help='batch size', default=8)
    parser.add_argument('--device', type=int, help='Device name', default=0)  
    parser.add_argument('--epochs', type=int, help='num of epochs', default=120)
    parser.add_argument('--dim', type=int, help='dim of input features', default=437)
    parser.add_argument('--lang', type=int, help='num of language classes', default=3)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--lambda', type=float, help='hyperparameter for joint training', default=0.5)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    # load model
    model = BLSTM_E2E_LID(n_lang=args.lang,
                          dropout=0.25,
                          input_dim=args.dim,
                          hidden_size=256,
                          num_emb_layer=2,
                          num_lstm_layer=3,
                          emb_dim=256)
    model.to(device)
    loss_func_DCL = DeepClusteringLoss().to(device)
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    # load data
    train_txt = args.train
    train_set = RawFeatures(train_txt)
    valid_txt = args.test
    valid_set = RawFeatures(valid_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=args.batch,
                            pin_memory=True,
                            num_workers=16,
                            shuffle=True,
                            collate_fn=collate_fn)

    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn)
    # optimizer & learning rate decay strategy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    T_max = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    # Train the model
    total_step = len(train_data)
    best_acc = 0

    for epoch in tqdm(range(args.epochs)):
        loss_item = 0
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            utt_ = rnn_utils.pack_padded_sequence(utt_, seq_len, batch_first=True)
            labels_ = rnn_util.pack_padded_sequence(labels, seq_len, batch_first=True).data.to(device)
            # Forward pass
            outputs, embeddings = model(utt_)
            loss_DCL = loss_func_DCL(embeddings, labels_)
            loss_CRE = loss_func_CRE(outputs, labels_)
            loss = args.lambda * loss_CRE + (1 - args.lambda) * loss_DCL
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} CRE: {:.4f} DCL: {:.4f}"
                      .format(epoch + 1, args.epochs, step + 1, total_step, loss.item(), loss_CRE.item(), loss_DCL.item()))
        scheduler.step()
        model.eval()
        correct = 0
        total = 0
        eer = 0
        FAR_list = torch.zeros(args.lang)
        FRR_list = torch.zeros(args.lang) 
        with torch.no_grad():
            for step, (utt, labels, seq_len) in enumerate(valid_data):
                utt = utt.to(device=device, dtype=torch.float)
                utt_ = rnn_utils.pack_padded_sequence(utt, seq_len, batch_first=True)
                labels_ = rnn_util.pack_padded_sequence(labels, seq_len, batch_first=True).data.to(device)
                outputs, embeddings = model(utt_)
                predicted = torch.argmax(outputs,-1)
                total += labels.size(-1)
                correct += (predicted == labels_).sum().item()
                FAR, FRR = compute_far_frr(args.lang, predicted, labels_)
                FAR_list += FAR
                FRR_list += FRR
            acc = correct / total
            print('Current Acc.: {:.4f} %'.format(100 * acc))
            for i in range(args.lang):
                eer_ = (FAR_list[i] / total + FRR_list[i] / total) / 2
                eer += eer_
                print("EER for label {}: {:.4f}%".format(i, eer_ * 100))
            print('EER: {:.4f} %'.format(100 * eer / args.lang))
        if acc > best_acc:
            print('New best Acc.: {:.4f}%, EER: {:.4f} %, model saved!'.format(100 * acc, 100 * eer / args.lang))
            best_acc = acc
            best_eer = eer / args.lang
            torch.save(model.state_dict(), '/home/hexin/Desktop/models/' + '{}.ckpt'.format(args.model))
    print('Final Acc: {:.4f}%, Final EER: {.4f}%'.format(100 * best_acc, 100 * best_eer))

if __name__ == "__main__":
    main()
