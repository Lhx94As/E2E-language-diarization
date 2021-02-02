import torch
import numpy as np

def compute_far_frr(num_lang, predicts, targets):
    labels = [i for i in range(num_lang)]
    T = targets.size(-1)
    EER = 0
    FRR = torch.zeros(num_lang)
    FAR = torch.zeros(num_lang)

    for i in labels:
        false_reject = 0
        false_alarm = 0
        for ii in range(predicts.size(0)):
            predict = predicts[ii]
            target = targets[ii]
            if target == i and predict != i:
                false_reject += 1
            elif target != i and predict == i:
                false_alarm += 1
        FAR[i] = false_alarm
        FRR[i] = false_reject
    #     eer_label = (false_alarm/T + false_reject/T)/2
    #     EER+=eer_label
    # EER = EER/num_spks
    return FAR, FRR


