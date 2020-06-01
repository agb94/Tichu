import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import random

from model import TichuNet1

def collate(state_rep_list, device='cuda'):
    ugh = [[] for _ in range(4)]
    for cr, ncr in state_rep_list:
        all_rep = cr + ncr
        for idx in range(4):
            ugh[idx].append(torch.LongTensor(all_rep[idx]))
    collated = [torch.cat(e, dim=0).to(device) for e in ugh]
    return collated[:2], collated[2:]
    print(collated[0])

def prep_data(data_list):
    input_data, target_data = zip(*data_list)
    input_tensors = collate(input_data, device=device)
    target_tensor = torch.Tensor(target_data).to(device).unsqueeze(1) / score_norm_factor
    return input_tensors, target_tensor
    
# parameters
device = 'cuda'
epoch_num = 300
data_path = './data/phase2.pkl'
learning_rate = 1e-3
batch_size = 100
score_norm_factor = 100.
val_size = 100

all_data = pkl.load(open(data_path, 'rb'))
listed_data = list(zip(all_data['input'], all_data['target']))

train_data = listed_data[:-val_size]
val_data = listed_data[-val_size:]
val_in, val_gt = prep_data(val_data)

net = TichuNet1()
net.load_state_dict(torch.load('./models/phase1_net.pth'))
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

for epoch in range(epoch_num):
    random.shuffle(train_data)
    for idx in range(0, len(train_data), batch_size):
        input_tensors, gt_output = prep_data(listed_data[idx:idx+batch_size])
        
        pred_output = net(*input_tensors)
        loss = torch.mean(torch.abs(gt_output - pred_output))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx/batch_size) % 100 == 99:
            with torch.no_grad():
                val_pred = net(*val_in)
                val_loss = torch.mean(torch.abs(val_pred - val_gt))
            print(f'[{idx//batch_size}/{epoch}] mean loss: {loss.item()}, val loss: {val_loss.item()}')
    
torch.save(net.state_dict(), 'models/phase2_net.pth')