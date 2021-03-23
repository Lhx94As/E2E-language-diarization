import os
import math
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from data_load import *
from model_evaluation import *

#python train_sa_e2e.py --savedir "/home/hexin/Desktop/models" --train "/home/hexin/Desktop/data/train.txt" --test "/home/hexin/Desktop/data/test.txt"
#                          --seed 0 --device 0 --batch 64 --epochs 30 --dim 23 --lang 3 --model my_xsa_model --lr 0.0001 --maxlength 666 --lambda 0.5 -- warmup 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
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
    parser.add_argument('--train', type=str, help='training data, in .txt')
    parser.add_argument('--test', type=str, help='testing data, in .txt')
    parser.add_argument('--savedir', type=str, help='dir in which the trained model is saved')
    parser.add_argument('--model', type=str, help='model name', default='my_sa_e2e')
    parser.add_argument('--seed', type=int, help='device id', default=0)
    parser.add_argument('--batch', type=int, help='batch size', default=64)
    parser.add_argument('--device', type=int, help='Device name', default=0)
    parser.add_argument('--warmup', type=int, help='num of epochs for warmup', default=10)
    parser.add_argument('--epochs', type=int, help='num of epochs in total', default=600)
    parser.add_argument('--dim', type=int, help='dim of input features', default=437)
    parser.add_argument('--lang', type=int, help='num of language classes', default=3)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.0001)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    # load model
    model = Transformer_E2E_LID(n_lang=args.lang,
                                dropout=0.1,
                                input_dim=args.dim,
                                feat_dim=256,
                                n_heads=4,
                                d_k=256,
                                d_v=256,
                                d_ff=2048,
                                max_seq_len=140,
                                device=device)
    model.to(device)
    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    
    # optimizer & warm up & learning rate decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup \
        if epoch <= args.warmup \
        else 0.5 * (math.cos((epoch - args.warmup) / (args.epochs - args.warmup) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    
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
                            collate_fn=collate_fn_atten)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    
    # Train the model
    total_step = len(train_data)
    best_acc = 0
    for epoch in tqdm(range(args.epochs)):
        # training stage
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            labels = labels.to(device=device, dtype=torch.long)
            # Forward pass
            outputs = model(utt_, seq_len, atten_mask)
            outputs = get_output(outputs, seq_len)
            loss = loss_func_CRE(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".
                      format(epoch + 1, args.epochs, step + 1, total_step, loss.item()))
        scheduler.step()
        print('Current LR: {}'.format(get_lr(optimizer)))
        # eval stage
        model.eval()
        eer = 0
        total = 0
        correct = 0 
        FAR_list = torch.zeros(args.lang)
        FRR_list = torch.zeros(args.lang)
        with torch.no_grad():
            for step, (utt, labels, seq_len) in enumerate(valid_data):
                utt_ = utt.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = model(x=utt_, seq_len=seq_len, atten_mask=None)
                outputs = get_output(outputs, seq_len)
                predicted = torch.argmax(outputs, -1)
                total += labels.size(-1)
                correct += (predicted == labels).sum().item()
                FAR, FRR = compute_far_frr(args.lang, predicted, labels)
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
            torch.save(model.state_dict(), args.savedir + '{}.ckpt'.format(args.model))
    print('Final Acc: {:.4f}%, Final EER: {.4f}%'.format(100 * best_acc, 100 * best_eer))


if __name__ == "__main__":
    main()
