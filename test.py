#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

def test(model, loader, device, CONFIG, metrics):
    print('bat dau test ne')
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        print('bat dau test trong metric ne')
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.propagate() 
        print('propagate model')
        for users, ground_truth_u_b, train_mask_u_b in loader:
            print('start evaluate')
            pred_b = model.evaluate(rs, users.to(device))  
            pred_b -= 1e8*train_mask_u_b.to(device)
            cnt = 0
            for metric in metrics:
                cnt += 1
                print('ab {}'.format(cnt))
                metric(pred_b, ground_truth_u_b.to(device))
                print('done')
            print('1')
    print('Test: time={:d}s'.format(int(time()-start)))
    for metric in metrics:
        print('xyz')
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    print('da test xong')
    return metrics

